# pro/minorbodycatalog.py
from __future__ import annotations

"""
Minor body (asteroid + comet) catalog helper for SASpro.

Responsibilities:
- Fetch JSON manifest from GitHub (saspro-minorbody-data repo).
- Download / update the SQLite minor body database from the release asset.
- Provide a small API for querying the DB and (optionally) computing
  RA/Dec for a subset of objects via Skyfield.

This module is intentionally independent of Qt; higher-level UI / QSettings
integration should live in the main app code and call into this helper.
"""

import json
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Tuple

# Skyfield imports (required for position computations)
from skyfield.api import load as sf_load
from skyfield.data import mpc as sf_mpc
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw GitHub URL to the manifest in your data repo
MANIFEST_URL = (
    "https://raw.githubusercontent.com/setiastro/"
    "saspro-minorbody-data/main/saspro_minor_bodies_manifest.json"
)

# Default filenames (as defined by the manifest you showed)
DEFAULT_DB_BASENAME = "saspro_minor_bodies.sqlite"
DEFAULT_MANIFEST_BASENAME = "saspro_minor_bodies_manifest.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MinorBodyManifest:
    schema_version: int
    version: str
    generated_utc: str
    download_url: str
    download_filename: str
    counts_asteroids: int
    counts_comets: int
    raw: Dict[str, Any]


# ---------------------------------------------------------------------------
# Network helpers (urllib only, to keep deps minimal)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """Fetch JSON from a URL using urllib."""
    req = urllib.request.Request(url, headers={"User-Agent": "SetiAstroSuitePro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _http_download_binary(url: str, dest: Path, chunk_size: int = 65536, timeout: float = 30.0) -> None:
    """Download a binary file from URL to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "SetiAstroSuitePro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f_out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
        tmp.replace(dest)


# ---------------------------------------------------------------------------
# Manifest + DB management
# ---------------------------------------------------------------------------

def fetch_remote_manifest(url: str = MANIFEST_URL) -> MinorBodyManifest:
    """Fetch the remote manifest from GitHub and parse it."""
    data = _http_get_json(url)
    # Defensive parsing: tolerate missing bits gracefully
    dl = data.get("download", {})
    counts = data.get("counts", {})

    return MinorBodyManifest(
        schema_version=int(data.get("schema_version", 1)),
        version=str(data.get("version", "unknown")),
        generated_utc=str(data.get("generated_utc", "")),
        download_url=str(dl.get("url", "")),
        download_filename=str(dl.get("filename", DEFAULT_DB_BASENAME)),
        counts_asteroids=int(counts.get("asteroids", 0)),
        counts_comets=int(counts.get("comets", 0)),
        raw=data,
    )


def load_local_manifest(path: Path) -> Optional[MinorBodyManifest]:
    """Load a previously saved local manifest (if it exists)."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dl = data.get("download", {})
        counts = data.get("counts", {})
        return MinorBodyManifest(
            schema_version=int(data.get("schema_version", 1)),
            version=str(data.get("version", "unknown")),
            generated_utc=str(data.get("generated_utc", "")),
            download_url=str(dl.get("url", "")),
            download_filename=str(dl.get("filename", DEFAULT_DB_BASENAME)),
            counts_asteroids=int(counts.get("asteroids", 0)),
            counts_comets=int(counts.get("comets", 0)),
            raw=data,
        )
    except Exception:
        return None


def save_local_manifest(path: Path, manifest: MinorBodyManifest) -> None:
    """Write manifest JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.raw, f, indent=2)


def ensure_minor_body_db(
    data_dir: Path,
    manifest_url: str = MANIFEST_URL,
    force_refresh: bool = False,
) -> Tuple[Path, MinorBodyManifest]:
    """
    Ensure that the minor body SQLite DB exists (and is up to date).

    Parameters
    ----------
    data_dir : Path
        Directory where the DB + local manifest should live.
    manifest_url : str
        URL to the JSON manifest in the saspro-minorbody-data repo.
    force_refresh : bool
        If True, always re-download the DB even if version matches.

    Returns
    -------
    db_path : Path
        Path to the local SQLite DB.
    manifest : MinorBodyManifest
        Parsed manifest object for the local version.

    Notes
    -----
    This function is network-only; any UI / progress should wrap it and
    catch exceptions.
    """
    data_dir = data_dir.resolve()
    local_manifest_path = data_dir / DEFAULT_MANIFEST_BASENAME

    remote = fetch_remote_manifest(manifest_url)
    db_path = data_dir / remote.download_filename

    local = load_local_manifest(local_manifest_path)

    needs_download = force_refresh

    if not needs_download:
        if local is None:
            needs_download = True
        elif local.version != remote.version:
            needs_download = True
        elif not db_path.is_file():
            needs_download = True

    if needs_download:
        if not remote.download_url:
            raise RuntimeError("Manifest does not contain a download URL for the DB.")
        _http_download_binary(remote.download_url, db_path)
        save_local_manifest(local_manifest_path, remote)
        manifest = remote
    else:
        # local manifest + DB are assumed valid
        manifest = local

    return db_path, manifest


# ---------------------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------------------

class MinorBodyCatalog:
    """
    Thin wrapper around the saspro minor body SQLite DB.

    Usage:
        cat = MinorBodyCatalog(db_path)
        print(cat.counts)
        df = cat.get_bright_asteroids(H_max=20.0, limit=50000)
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Minor body DB not found: {self.db_path}")
        # We keep connection lazy; open on demand
        self._conn: Optional[sqlite3.Connection] = None

    # ---- Connection management -------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Use read-only URI mode when possible
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- Introspection ---------------------------------------------------------

    @property
    def counts(self) -> Dict[str, int]:
        """Return row counts for 'asteroids' and 'comets' tables."""
        conn = self._get_conn()
        cur = conn.cursor()
        result = {}
        for table in ("asteroids", "comets"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                result[table] = int(cur.fetchone()[0])
            except sqlite3.Error:
                result[table] = 0
        return result

    # ---- Simple queries --------------------------------------------------------

    def get_bright_asteroids(
        self,
        H_max: float = 20.0,
        limit: Optional[int] = 100000,
    ):
        """
        Return a pandas DataFrame of relatively bright asteroids.

        Parameters
        ----------
        H_max : float
            Only include asteroids with absolute magnitude H <= H_max.
            (Column is 'magnitude_H' in the Skyfield-generated table.)
        limit : int or None
            Maximum number of rows to return; None for no limit.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        conn = self._get_conn()
        sql = """
            SELECT *
            FROM asteroids
            WHERE magnitude_H <= ?
            ORDER BY magnitude_H ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            df = pd.read_sql_query(sql, conn, params=(H_max, limit))
        else:
            df = pd.read_sql_query(sql, conn, params=(H_max,))
        return df

    def get_bright_comets(
        self,
        H_max: float = 15.0,
        limit: Optional[int] = 5000,
    ):
        """
        Return a pandas DataFrame of 'bright' comets.

        The magnitude columns in the comets table differ slightly;
        Skyfield usually stores them as 'absolute_magnitude' (or similar).
        This function uses a best-effort filter and is mainly a convenience.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        conn = self._get_conn()
        # Try to detect magnitude column
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(comets)")
        cols = {row[1] for row in cur.fetchall()}

        mag_col = None
        for candidate in ("absolute_magnitude", "magnitude_H", "H"):
            if candidate in cols:
                mag_col = candidate
                break

        if mag_col is None:
            # No magnitude info, just return a limited subset
            sql = "SELECT * FROM comets"
            if limit is not None:
                sql += " LIMIT ?"
                return pd.read_sql_query(sql, conn, params=(limit,))
            return pd.read_sql_query(sql, conn)

        sql = f"""
            SELECT *
            FROM comets
            WHERE {mag_col} <= ?
            ORDER BY {mag_col} ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            df = pd.read_sql_query(sql, conn, params=(H_max, limit))
        else:
            df = pd.read_sql_query(sql, conn, params=(H_max,))
        return df

    def get_asteroids_by_designation(
        self,
        designations: Iterable[str],
    ):
        """
        Fetch asteroid rows for one or more MPC designations.

        Parameters
        ----------
        designations : iterable of str
            MPC designations like "1 Ceres", "433 Eros", etc.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        designations = list(designations)
        if not designations:
            import pandas as pd  # type: ignore
            return pd.DataFrame()

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in designations)
        sql = f"""
            SELECT *
            FROM asteroids
            WHERE designation IN ({placeholders})
        """
        return pd.read_sql_query(sql, conn, params=designations)

    # -----------------------------------------------------------------------
    # Ephemeris / position scaffold (Skyfield-based, for small subsets)
    # -----------------------------------------------------------------------
    def compute_positions_skyfield(
        self,
        asteroid_rows,
        jd: float,
        ephemeris_path: Optional[Path] = None,
        topocentric: Optional[Tuple[float, float, float]] = None,
        debug: bool = False,
    ):
        import pandas as pd

        if isinstance(asteroid_rows, pd.DataFrame):
            df = asteroid_rows.copy()
        else:
            df = pd.DataFrame(list(asteroid_rows))

        if debug:
            print("[MinorBodies] DataFrame columns:", list(df.columns))

        # REQUIRED NUMERIC COLUMNS (note: epoch_packed is deliberately *not* here)
        required_numeric = [
            "mean_anomaly_degrees",
            "argument_of_perihelion_degrees",
            "longitude_of_ascending_node_degrees",
            "inclination_degrees",
            "eccentricity",
            "mean_daily_motion_degrees",
            "semimajor_axis_au",
        ]

        for col in required_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)
        df = df.dropna(subset=[c for c in required_numeric if c in df.columns])
        if debug:
            print(
                f"[MinorBodies] rows after dropping NaNs in required numeric cols: "
                f"{len(df)} (was {before})"
            )
            # Optional: peek at first few numeric rows for sanity
            try:
                print("[MinorBodies] sample numeric rows:\n",
                      df[required_numeric].head(3))
            except Exception:
                pass

        if df.empty:
            if debug:
                print("[MinorBodies] no valid rows after cleaning; aborting.")
            return []

        ts = sf_load.timescale()
        t = ts.tt_jd(jd)

        if ephemeris_path is not None:
            eph = sf_load(str(ephemeris_path))
        else:
            eph = sf_load("de440s.bsp")

        sun = eph["sun"]
        earth = eph["earth"]

        if topocentric is not None:
            from skyfield.api import wgs84
            lat_deg, lon_deg, elev_m = topocentric
            earth = earth + wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)

        results = []
        total = len(df)
        ok = 0
        failed = 0

        for idx, row in df.iterrows():
            try:
                orb = sf_mpc.mpcorb_orbit(row, ts, GM_SUN)
                body = sun + orb  # Sun-centered orbit so Earth can observe it
                ast_at_t = earth.at(t).observe(body).apparent()
                ra, dec, distance = ast_at_t.radec()

                results.append(
                    {
                        "designation": row.get("designation", ""),
                        "ra_deg": float(ra.hours * 15.0),
                        "dec_deg": float(dec.degrees),
                        "distance_au": float(distance.au),
                    }
                )
                ok += 1
            except Exception as e:
                failed += 1
                if debug and failed <= 10:
                    print(
                        f"[MinorBodies] mpcorb_orbit/observe FAILED for "
                        f"'{row.get('designation', '')}': {repr(e)}"
                    )

        if debug:
            print(
                f"[MinorBodies] Skyfield positions: total={total}, ok={ok}, failed={failed}"
            )

        return results
