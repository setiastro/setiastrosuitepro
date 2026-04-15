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

import numpy as np

# Astropy imports for local Kepler propagation
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
import astropy.units as u

# Skyfield imports kept for backward compatibility but no longer used
# for position computation — can be removed entirely if desired
try:
    from skyfield.api import load as sf_load
    from skyfield.data import mpc as sf_mpc
    from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
except Exception:
    sf_load = None
    sf_mpc  = None
    GM_SUN  = None
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

def _solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for E (eccentric anomaly).
    Uses Newton-Raphson iteration.
    """
    # Good initial guess
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _decode_packed_epoch(packed: str) -> float:
    """
    Decode MPC packed epoch string (e.g. 'K245N') to Julian date.
    Format: C YY MM DD where C is century letter, MM/DD are base-36 encoded.
    """
    if not packed or len(packed) < 5:
        # fallback to J2000
        return 2451545.0

    def n(c):
        return ord(c) - (48 if c.isdigit() else 55)

    try:
        century = 100 * n(packed[0]) + int(packed[1:3])
        month   = n(packed[3])
        day     = n(packed[4])

        # Julian day from calendar date (same formula as Skyfield uses)
        y = century
        m = month
        d = day
        if m <= 2:
            y -= 1
            m += 12
        A = int(y / 100)
        B = 2 - A + int(A / 4)
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
        return jd
    except Exception:
        return 2451545.0
    
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

        We treat magnitude_H as numeric even though the column is TEXT in the
        current schema, by casting it to REAL for filtering and ordering.
        """
        import pandas as pd

        conn = self._get_conn()

        # Force numeric comparison + ordering
        sql = """
            SELECT *
            FROM asteroids
            WHERE CAST(magnitude_H AS REAL) <= ?
            ORDER BY CAST(magnitude_H AS REAL) ASC
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

        We auto-detect the magnitude column and cast it to REAL so that
        filtering and ordering are done in brightness order.
        """
        import pandas as pd

        conn = self._get_conn()
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

        # Cast to REAL so TEXT columns behave numerically
        mag_expr = f"CAST({mag_col} AS REAL)"

        sql = f"""
            SELECT *
            FROM comets
            WHERE {mag_expr} <= ?
            ORDER BY {mag_expr} ASC
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
        import pandas as pd

        conn = self._get_conn()
        # Try to detect magnitude column
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(comets)")
        cols = {row[1] for row in cur.fetchall()}

        mag_col = None
        # Prefer g, then k, then any H-style absolute mag
        for candidate in ("magnitude_g", "magnitude_k", "absolute_magnitude", "magnitude_H", "H"):
            if candidate in cols:
                mag_col = candidate
                break

        if mag_col is None:
            # No magnitude info we recognize – just return a limited subset
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
    # Ephemeris / position scaffold (astropy-based, for small subsets)
    # -----------------------------------------------------------------------
    def compute_positions_astropy(
        self,
        asteroid_rows,
        jd: float,
        ephemeris_path=None,
        topocentric=None,
        progress_cb=None,
        debug: bool = False,
    ):
        """
        Compute RA/Dec using astropy + local orbital elements only.
        No network calls, no Skyfield frame issues.
        """
        import pandas as pd
        import numpy as np
        from astropy.time import Time
        from astropy.coordinates import (
            SkyCoord, GCRS, HeliocentricEclipticIAU76,
            get_body_barycentric_posvel, CartesianRepresentation
        )
        import astropy.units as u

        if isinstance(asteroid_rows, pd.DataFrame):
            df = asteroid_rows.copy()
        else:
            df = pd.DataFrame(list(asteroid_rows))

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
            print(f"[MinorBodies] rows after cleaning: {len(df)} (was {before})")

        if df.empty:
            return []

        # Observation time
        t_obs = Time(jd, format="jd", scale="tt")
        if debug:
            print(f"[MinorBodies] obs time: {t_obs.iso}  JD={jd:.6f}")

        # Earth barycentric position at observation time (ICRS, AU)
        from astropy.coordinates import solar_system_ephemeris
        with solar_system_ephemeris.set("builtin"):
            earth_bary, _ = get_body_barycentric_posvel("earth", t_obs)
        earth_xyz = np.array([
            earth_bary.x.to(u.au).value,
            earth_bary.y.to(u.au).value,
            earth_bary.z.to(u.au).value,
        ])

        # Sun barycentric position
        with solar_system_ephemeris.set("builtin"):
            sun_bary, _ = get_body_barycentric_posvel("sun", t_obs)
        sun_xyz = np.array([
            sun_bary.x.to(u.au).value,
            sun_bary.y.to(u.au).value,
            sun_bary.z.to(u.au).value,
        ])

        if debug:
            print(f"[MinorBodies] Earth ICRS (AU): {earth_xyz}")

        results = []
        total = len(df)
        ok = 0
        failed = 0

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if progress_cb is not None:
                try:
                    cont = progress_cb(i, total)
                except Exception:
                    cont = True
                if cont is False:
                    if debug:
                        print("[MinorBodies] aborted by progress_cb")
                    break

            try:
                # ── 1) Decode epoch ───────────────────────────────────────────
                epoch_packed = str(row.get("epoch_packed", "")).strip()
                epoch_jd = _decode_packed_epoch(epoch_packed)
                t_epoch = Time(epoch_jd, format="jd", scale="tt")

                # ── 2) Orbital elements ───────────────────────────────────────
                a   = float(row["semimajor_axis_au"])          # AU
                e   = float(row["eccentricity"])
                inc = np.radians(float(row["inclination_degrees"]))
                Om  = np.radians(float(row["longitude_of_ascending_node_degrees"]))
                om  = np.radians(float(row["argument_of_perihelion_degrees"]))
                M0  = np.radians(float(row["mean_anomaly_degrees"]))
                n   = np.radians(float(row["mean_daily_motion_degrees"]))  # rad/day

                # ── 3) Propagate mean anomaly to obs time ─────────────────────
                dt_days = float((t_obs - t_epoch).jd)
                M = M0 + n * dt_days
                M = M % (2 * np.pi)

                # ── 4) Solve Kepler's equation  M = E - e*sin(E) ─────────────
                E = _solve_kepler(M, e)

                # ── 5) True anomaly ───────────────────────────────────────────
                cos_E = np.cos(E)
                sin_E = np.sin(E)
                nu = np.arctan2(
                    np.sqrt(1 - e*e) * sin_E,
                    cos_E - e
                )

                # ── 6) Heliocentric distance ──────────────────────────────────
                r = a * (1 - e * cos_E)

                # ── 7) Position in orbital plane ──────────────────────────────
                x_orb = r * np.cos(nu)
                y_orb = r * np.sin(nu)

                # ── 8) Rotate to ecliptic J2000 ───────────────────────────────
                cos_Om = np.cos(Om); sin_Om = np.sin(Om)
                cos_om = np.cos(om); sin_om = np.sin(om)
                cos_i  = np.cos(inc); sin_i = np.sin(inc)

                # Standard rotation matrix: orbital → ecliptic
                Xx =  cos_Om*cos_om - sin_Om*sin_om*cos_i
                Xy = -cos_Om*sin_om - sin_Om*cos_om*cos_i
                Yx =  sin_Om*cos_om + cos_Om*sin_om*cos_i
                Yy = -sin_Om*sin_om + cos_Om*cos_om*cos_i
                Zx =  sin_om*sin_i
                Zy =  cos_om*sin_i

                x_ecl = Xx*x_orb + Xy*y_orb
                y_ecl = Yx*x_orb + Yy*y_orb
                z_ecl = Zx*x_orb + Zy*y_orb

                # ── 9) Ecliptic J2000 → ICRS (rotate by obliquity of J2000) ──
                eps = np.radians(23.439291111)   # obliquity J2000.0
                cos_e = np.cos(eps); sin_e = np.sin(eps)

                x_icrs = x_ecl
                y_icrs = cos_e*y_ecl - sin_e*z_ecl
                z_icrs = sin_e*y_ecl + cos_e*z_ecl

                # ── 10) Heliocentric ICRS → barycentric ICRS ──────────────────
                # Add Sun's barycentric position
                x_bary = x_icrs + sun_xyz[0]
                y_bary = y_icrs + sun_xyz[1]
                z_bary = z_icrs + sun_xyz[2]

                # ── 11) Topocentric vector (asteroid − Earth) ─────────────────
                dx = x_bary - earth_xyz[0]
                dy = y_bary - earth_xyz[1]
                dz = z_bary - earth_xyz[2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                # ── 12) ICRS RA/Dec ───────────────────────────────────────────
                sc = SkyCoord(
                    x=dx*u.au, y=dy*u.au, z=dz*u.au,
                    representation_type="cartesian",
                    frame="icrs",
                )
                sc_sph = sc.represent_as("unitspherical")

                results.append({
                    "designation": row.get("designation", ""),
                    "ra_deg":      float(sc_sph.lon.deg),
                    "dec_deg":     float(sc_sph.lat.deg),
                    "distance_au": float(dist),
                })
                ok += 1

            except Exception as exc:
                failed += 1
                if debug and failed <= 10:
                    print(f"[MinorBodies] FAILED '{row.get('designation','')}': {repr(exc)}")

        if debug:
            print(f"[MinorBodies] total={total}, ok={ok}, failed={failed}")

        return results