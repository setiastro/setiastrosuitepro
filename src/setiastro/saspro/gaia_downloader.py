# gaia_downloader.py
# Downloads Gaia DR3 XP spectra and creates our own local database

from __future__ import annotations

import os
import sqlite3
import zlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np
import hashlib
import time
try:
    from astroquery.gaia import Gaia
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    # gaiaxpy API varies slightly by version, but calibrate() exists across releases
    from gaiaxpy import calibrate
    HAS_GAIAXPY = True
except ImportError:
    HAS_GAIAXPY = False


@dataclass
class GaiaSource:
    """A Gaia source with basic parameters."""
    source_id: int
    ra: float
    dec: float
    phot_g_mean_mag: Optional[float] = None
    bp_rp: Optional[float] = None
    parallax: Optional[float] = None
    pmra: Optional[float] = None
    pmdec: Optional[float] = None


@dataclass
class CalibratedSpectrum:
    """A calibrated XP spectrum."""
    source_id: int
    wavelengths: np.ndarray  # nm
    flux: np.ndarray  # W/nm/m^2 (per GaiaXPy)
    flux_error: Optional[np.ndarray] = None

    @property
    def normalized_flux(self) -> np.ndarray:
        """Flux normalized to 0-1 range."""
        m = float(np.max(self.flux)) if self.flux is not None and self.flux.size else 0.0
        if m > 0:
            return self.flux / m
        return self.flux

    def get_flux_at(self, wavelength_nm: float) -> float:
        """Interpolate flux at specific wavelength."""
        return float(np.interp(wavelength_nm, self.wavelengths, self.flux))


class GaiaSpectraDB:
    """
    Local database for storing Gaia XP spectra.

    Uses SQLite for metadata and compressed numpy arrays for spectra.
    Optimized for SFCC-like use (fast lookup + compact).
    """

    # Standard wavelength grid (GaiaXPy default: 336..1020 nm, step 2 nm, inclusive)
    WAVELENGTH_START = 336  # nm
    WAVELENGTH_END = 1020   # nm
    WAVELENGTH_STEP = 2     # nm
    N_WAVELENGTHS = int((WAVELENGTH_END - WAVELENGTH_START) / WAVELENGTH_STEP) + 1  # 343

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")

        cursor = self._conn.cursor()

        # Sources table - basic Gaia parameters
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY,
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                phot_g_mean_mag REAL,
                bp_rp REAL,
                parallax REAL,
                pmra REAL,
                pmdec REAL,
                has_xp_spectrum INTEGER DEFAULT 0
            )
        ''')

        # Spectra table - compressed flux arrays
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectra (
                source_id INTEGER PRIMARY KEY,
                flux_compressed BLOB NOT NULL,
                flux_error_compressed BLOB,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        ''')
        # Synthetic photometry cache:
        # stores integrals of Gaia flux * system throughput for a given system_key
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synth_phot (
                source_id INTEGER NOT NULL,
                system_key TEXT NOT NULL,
                Sr REAL NOT NULL,
                Sg REAL NOT NULL,
                Sb REAL NOT NULL,
                created_utc INTEGER,
                PRIMARY KEY (source_id, system_key),
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_synth_phot_system
            ON synth_phot(system_key)
        ''')
        # Metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # Store wavelength grid info
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value) VALUES
            ('wavelength_start', ?),
            ('wavelength_end', ?),
            ('wavelength_step', ?),
            ('n_wavelengths', ?)
        ''', (self.WAVELENGTH_START, self.WAVELENGTH_END,
              self.WAVELENGTH_STEP, self.N_WAVELENGTHS))

        # Create indices
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sources_coords
            ON sources(ra, dec)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sources_mag
            ON sources(phot_g_mean_mag)
        ''')

        self._conn.commit()

    @property
    def wavelengths(self) -> np.ndarray:
        """Standard wavelength grid (exact 2nm steps, inclusive)."""
        return np.arange(
            self.WAVELENGTH_START,
            self.WAVELENGTH_END + self.WAVELENGTH_STEP,
            self.WAVELENGTH_STEP,
            dtype=np.float32
        )

    def _compress_array(self, arr: np.ndarray) -> bytes:
        """Compress numpy array to bytes."""
        arr_f32 = np.asarray(arr, dtype=np.float32).reshape(-1)
        return zlib.compress(arr_f32.tobytes(), level=6)

    def _decompress_array(self, data: bytes) -> np.ndarray:
        """Decompress bytes to numpy array."""
        decompressed = zlib.decompress(data)
        arr = np.frombuffer(decompressed, dtype=np.float32)
        return arr

    def add_source(self, source: GaiaSource):
        """Add a source to the database."""
        cursor = self._conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sources
            (source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(source.source_id),
            float(source.ra),
            float(source.dec),
            float(source.phot_g_mean_mag) if source.phot_g_mean_mag is not None else None,
            float(source.bp_rp) if source.bp_rp is not None else None,
            float(source.parallax) if source.parallax is not None else None,
            float(source.pmra) if source.pmra is not None else None,
            float(source.pmdec) if source.pmdec is not None else None,
        ))
        self._conn.commit()

    def add_spectra(self, spectra: List[CalibratedSpectrum]) -> int:
        """
        Bulk insert spectra into database.
        Returns number successfully stored.
        """
        if not spectra:
            return 0

        cursor = self._conn.cursor()
        count = 0

        # Use a single transaction for speed
        try:
            cursor.execute("BEGIN;")
            for spectrum in spectra:
                # Reuse the single-spectrum logic but avoid committing each time:
                self._add_spectrum_no_commit(cursor, spectrum)
                count += 1
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return count


    def _add_spectrum_no_commit(self, cursor: sqlite3.Cursor, spectrum: CalibratedSpectrum):
        """Internal helper: add one spectrum using an existing cursor/transaction (no commit)."""
        wl_std = self.wavelengths

        if spectrum.wavelengths is None or len(spectrum.wavelengths) != self.N_WAVELENGTHS:
            wl_in = np.asarray(spectrum.wavelengths, dtype=np.float32).reshape(-1)
            flux_in = np.asarray(spectrum.flux, dtype=np.float32).reshape(-1)
            flux_resampled = np.interp(wl_std, wl_in, flux_in, left=0.0, right=0.0).astype(np.float32)

            if spectrum.flux_error is not None:
                err_in = np.asarray(spectrum.flux_error, dtype=np.float32).reshape(-1)
                err_resampled = np.interp(wl_std, wl_in, err_in, left=0.0, right=0.0).astype(np.float32)
            else:
                err_resampled = None
        else:
            flux_resampled = np.asarray(spectrum.flux, dtype=np.float32).reshape(-1)
            err_resampled = (
                np.asarray(spectrum.flux_error, dtype=np.float32).reshape(-1)
                if spectrum.flux_error is not None else None
            )

        if flux_resampled.size != self.N_WAVELENGTHS:
            raise ValueError(f"Flux length {flux_resampled.size} != expected {self.N_WAVELENGTHS}")
        if err_resampled is not None and err_resampled.size != self.N_WAVELENGTHS:
            raise ValueError(f"Flux error length {err_resampled.size} != expected {self.N_WAVELENGTHS}")

        flux_blob = self._compress_array(flux_resampled)
        err_blob = self._compress_array(err_resampled) if err_resampled is not None else None

        cursor.execute('''
            INSERT OR REPLACE INTO spectra (source_id, flux_compressed, flux_error_compressed)
            VALUES (?, ?, ?)
        ''', (int(spectrum.source_id), flux_blob, err_blob))

        cursor.execute('''
            UPDATE sources SET has_xp_spectrum = 1 WHERE source_id = ?
        ''', (int(spectrum.source_id),))


    def add_spectrum(self, spectrum: CalibratedSpectrum):
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN;")
            self._add_spectrum_no_commit(cursor, spectrum)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


    def get_spectrum(self, source_id: int) -> Optional[CalibratedSpectrum]:
        """Retrieve a spectrum by source ID."""
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT flux_compressed, flux_error_compressed FROM spectra
            WHERE source_id = ?
        ''', (int(source_id),))

        row = cursor.fetchone()
        if row is None:
            return None

        flux = self._decompress_array(row[0])
        flux_error = self._decompress_array(row[1]) if row[1] else None

        # ensure correct length (older DBs / corruption)
        if flux.size != self.N_WAVELENGTHS:
            flux = flux[:self.N_WAVELENGTHS]
        if flux_error is not None and flux_error.size != self.N_WAVELENGTHS:
            flux_error = flux_error[:self.N_WAVELENGTHS]

        return CalibratedSpectrum(
            source_id=int(source_id),
            wavelengths=self.wavelengths,
            flux=flux,
            flux_error=flux_error
        )

    def get_source(self, source_id: int) -> Optional[GaiaSource]:
        """Retrieve source info by ID."""
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec
            FROM sources WHERE source_id = ?
        ''', (int(source_id),))

        row = cursor.fetchone()
        if row is None:
            return None

        return GaiaSource(
            source_id=int(row[0]),
            ra=float(row[1]),
            dec=float(row[2]),
            phot_g_mean_mag=float(row[3]) if row[3] is not None else None,
            bp_rp=float(row[4]) if row[4] is not None else None,
            parallax=float(row[5]) if row[5] is not None else None,
            pmra=float(row[6]) if row[6] is not None else None,
            pmdec=float(row[7]) if row[7] is not None else None,
        )
    @staticmethod
    def make_system_key(
        wl_grid_ang: np.ndarray,
        T_sys_R: np.ndarray,
        T_sys_G: np.ndarray,
        T_sys_B: np.ndarray,
        *,
        tag: str = "spcc_v1",
    ) -> str:
        """
        Build a stable cache key for a particular throughput set + wavelength grid.
        - wl_grid_ang: Å grid you integrate on (float array)
        - T_sys_*: throughput arrays on that same grid
        - tag: bump when you change integration logic / normalization, etc.
        """
        h = hashlib.sha1()
        h.update(tag.encode("utf-8"))

        for arr in (wl_grid_ang, T_sys_R, T_sys_G, T_sys_B):
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            h.update(a.tobytes())

        return h.hexdigest()

    def get_synth_phot(
        self,
        source_ids: List[int],
        system_key: str,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Fetch cached synthetic photometry for many sources.
        Returns dict source_id -> (Sr,Sg,Sb) for those found.
        """
        if not source_ids:
            return {}

        ids = [int(x) for x in source_ids]
        out: Dict[int, Tuple[float, float, float]] = {}

        # chunk the IN() list to avoid SQLite parameter limits
        CHUNK = 900
        cur = self._conn.cursor()

        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))

            cur.execute(
                f"""
                SELECT source_id, Sr, Sg, Sb
                FROM synth_phot
                WHERE system_key = ?
                AND source_id IN ({qmarks})
                """,
                [str(system_key)] + chunk,
            )

            for sid, Sr, Sg, Sb in cur.fetchall():
                out[int(sid)] = (float(Sr), float(Sg), float(Sb))

        return out

    def upsert_synth_phot(
        self,
        rows: List[Tuple[int, float, float, float]],
        system_key: str,
        *,
        created_utc: Optional[int] = None,
    ) -> int:
        """
        Bulk upsert synthetic photometry rows.
        rows: [(source_id, Sr, Sg, Sb), ...]
        Returns number written.
        """
        if not rows:
            return 0

        ts = int(created_utc) if created_utc is not None else int(time.time())
        cur = self._conn.cursor()

        try:
            cur.execute("BEGIN;")
            cur.executemany(
                """
                INSERT INTO synth_phot (source_id, system_key, Sr, Sg, Sb, created_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, system_key) DO UPDATE SET
                    Sr=excluded.Sr,
                    Sg=excluded.Sg,
                    Sb=excluded.Sb,
                    created_utc=excluded.created_utc
                """,
                [(int(sid), str(system_key), float(Sr), float(Sg), float(Sb), ts) for (sid, Sr, Sg, Sb) in rows],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return len(rows)

    def delete_synth_phot_system(self, system_key: str) -> int:
        """Remove cached integrals for a given system_key (useful for debugging)."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM synth_phot WHERE system_key = ?", (str(system_key),))
        n = cur.rowcount
        self._conn.commit()
        return int(n)

    @staticmethod
    def _wrap_ra(ra_deg: float) -> float:
        r = float(ra_deg) % 360.0
        if r < 0:
            r += 360.0
        return r

    def query_region(self, ra: float, dec: float, radius_deg: float,
                    mag_limit: Optional[float] = None,
                    only_with_spectra: bool = False) -> List[GaiaSource]:
        """Query sources in a circular region."""
        cursor = self._conn.cursor()

        ra = self._wrap_ra(ra)
        dec = float(dec)
        radius_deg = float(radius_deg)

        # Small-angle box prefilter
        cosd = max(1e-6, float(np.cos(np.radians(dec))))
        dra = radius_deg / cosd

        ra_min = self._wrap_ra(ra - dra)
        ra_max = self._wrap_ra(ra + dra)
        dec_min = dec - radius_deg
        dec_max = dec + radius_deg

        query = '''
            SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec
            FROM sources
            WHERE dec BETWEEN ? AND ?
        '''
        params: List[object] = [dec_min, dec_max]

        # RA wrap handling (0/360 crossing)
        if ra_min <= ra_max:
            query += ' AND ra BETWEEN ? AND ?'
            params.extend([ra_min, ra_max])
        else:
            query += ' AND (ra >= ? OR ra <= ?)'
            params.extend([ra_min, ra_max])

        if mag_limit is not None:
            query += ' AND phot_g_mean_mag <= ?'
            params.append(float(mag_limit))

        if only_with_spectra:
            query += ' AND has_xp_spectrum = 1'

        cursor.execute(query, params)

        sources: List[GaiaSource] = []
        for row in cursor.fetchall():
            # Verify actual distance (simple planar approximation; OK for small radii)
            rra = float(row[1])
            rdec = float(row[2])

            # shortest RA difference across wrap
            d_ra = (rra - ra + 540.0) % 360.0 - 180.0
            d_dec = (rdec - dec)
            d = np.sqrt((d_ra * cosd) ** 2 + (d_dec) ** 2)
            if d <= radius_deg:
                sources.append(GaiaSource(
                    source_id=int(row[0]),
                    ra=float(row[1]),
                    dec=float(row[2]),
                    phot_g_mean_mag=float(row[3]) if row[3] is not None else None,
                    bp_rp=float(row[4]) if row[4] is not None else None,
                    parallax=float(row[5]) if row[5] is not None else None,
                    pmra=float(row[6]) if row[6] is not None else None,
                    pmdec=float(row[7]) if row[7] is not None else None,
                ))

        return sources

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self._conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM sources')
        n_sources = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM spectra')
        n_spectra = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM sources WHERE has_xp_spectrum = 1')
        n_with_spectra = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(phot_g_mean_mag), MAX(phot_g_mean_mag) FROM sources')
        mag_range = cursor.fetchone()

        cursor.execute('SELECT MIN(ra), MAX(ra) FROM sources')
        ra_range = cursor.fetchone()

        cursor.execute('SELECT MIN(dec), MAX(dec) FROM sources')
        dec_range = cursor.fetchone()

        return {
            'total_sources': n_sources,
            'total_spectra': n_spectra,
            'sources_with_spectra': n_with_spectra,
            'mag_range': mag_range,
            'ra_range': ra_range,
            'dec_range': dec_range,
            'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        stats = self.get_stats()
        return f"GaiaSpectraDB('{self.db_path.name}', sources={stats['total_sources']}, spectra={stats['total_spectra']})"


class GaiaDownloader:
    """
    Downloads Gaia DR3 data and XP spectra from the archive.

    Usage:
        downloader = GaiaDownloader(db_path="my_gaia_spectra.db")

        downloader.download_region("M31", radius_arcmin=10, mag_limit=15)

        downloader.download_region_coords(ra=10.68, dec=41.27, radius_deg=0.5)
    """

    def __init__(self, db_path: Union[str, Path] = "gaia_spectra.db"):
        if not HAS_ASTROQUERY:
            raise ImportError("astroquery is required. Install with: pip install astroquery")
        if not HAS_ASTROPY:
            raise ImportError("astropy is required. Install with: pip install astropy")

        self.db = GaiaSpectraDB(db_path)
        self._check_gaiaxpy()

    def _check_gaiaxpy(self):
        """Check if GaiaXPy is available."""
        if not HAS_GAIAXPY:
            print("Warning: GaiaXPy not installed. XP spectra calibration will not be available.")
            print("Install with: pip install gaiaxpy")

    def resolve_target(self, target_name: str) -> Tuple[float, float]:
        """Resolve target name to coordinates using name resolver (via astropy)."""
        try:
            coord = SkyCoord.from_name(target_name)
            return coord.ra.deg, coord.dec.deg
        except Exception as e:
            raise ValueError(f"Could not resolve target '{target_name}': {e}")

    def query_gaia_sources(self, ra: float, dec: float, radius_deg: float,
                           mag_limit: float = 17.0,
                           max_sources: int = 10000) -> List[GaiaSource]:
        """
        Query Gaia DR3 for sources in a region that have XP continuous spectra.
        """
        ra = float(ra)
        dec = float(dec)
        radius_deg = float(radius_deg)
        mag_limit = float(mag_limit)
        max_sources = int(max_sources)

        # ADQL query for Gaia DR3
        # NOTE: has_xp_continuous is boolean in the archive; use true/false (not 'True')
        query = f"""
        SELECT TOP {max_sources}
            source_id,
            ra,
            dec,
            phot_g_mean_mag,
            bp_rp,
            parallax,
            pmra,
            pmdec
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        )
        AND phot_g_mean_mag < {mag_limit}
        AND has_xp_continuous = true
        ORDER BY phot_g_mean_mag ASC
        """

        print(f"Querying Gaia DR3 for sources within {radius_deg}° of ({ra:.4f}, {dec:.4f})...")

        job = Gaia.launch_job_async(query)
        result = job.get_results()

        sources: List[GaiaSource] = []
        for row in result:
            sources.append(GaiaSource(
                source_id=int(row['source_id']),
                ra=float(row['ra']),
                dec=float(row['dec']),
                phot_g_mean_mag=float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] is not None else None,
                bp_rp=float(row['bp_rp']) if row['bp_rp'] is not None else None,
                parallax=float(row['parallax']) if row['parallax'] is not None else None,
                pmra=float(row['pmra']) if row['pmra'] is not None else None,
                pmdec=float(row['pmdec']) if row['pmdec'] is not None else None,
            ))

        print(f"Found {len(sources)} sources with XP spectra")
        return sources

    @staticmethod
    def _sampling_to_wavelengths(sampling, fallback: np.ndarray) -> np.ndarray:
        """
        GaiaXPy 'sampling' differs by version:
          - sometimes iterable of floats
          - sometimes has attribute .wavelength (array-like)
          - sometimes has .values or similar
        """
        if sampling is None:
            return fallback
        for attr in ("wavelength", "wavelengths", "values"):
            if hasattr(sampling, attr):
                try:
                    w = np.asarray(getattr(sampling, attr), dtype=np.float32).reshape(-1)
                    if w.size:
                        return w
                except Exception:
                    pass
        # final attempt: treat as iterable
        try:
            w = np.asarray(list(sampling), dtype=np.float32).reshape(-1)
            if w.size:
                return w
        except Exception:
            pass
        return fallback

    def download_xp_spectra(self, source_ids: List[int],
                            batch_size: int = 100) -> List[CalibratedSpectrum]:
        """
        Download and calibrate XP spectra for given source IDs.
        """
        if not HAS_GAIAXPY:
            raise ImportError("GaiaXPy is required for spectrum calibration")

        all_spectra: List[CalibratedSpectrum] = []
        fallback_wl = self.db.wavelengths  # 336..1020 step 2

        for i in range(0, len(source_ids), batch_size):
            batch_ids = [int(x) for x in source_ids[i:i + batch_size]]
            print(f"Downloading spectra {i+1}-{min(i+batch_size, len(source_ids))} of {len(source_ids)}...")

            try:
                calibrated, sampling = calibrate(batch_ids)

                wavelengths = self._sampling_to_wavelengths(sampling, fallback=fallback_wl)

                # GaiaXPy returns a DataFrame with per-source 'flux' and (often) 'flux_error'
                for _, row in calibrated.iterrows():
                    flux = np.asarray(row['flux'], dtype=np.float32).reshape(-1)
                    flux_error = None
                    if 'flux_error' in calibrated.columns and row.get('flux_error') is not None:
                        try:
                            flux_error = np.asarray(row['flux_error'], dtype=np.float32).reshape(-1)
                        except Exception:
                            flux_error = None

                    all_spectra.append(CalibratedSpectrum(
                        source_id=int(row['source_id']),
                        wavelengths=wavelengths,
                        flux=flux,
                        flux_error=flux_error
                    ))

            except Exception as e:
                print(f"Warning: Failed to download batch: {e}")
                continue

        return all_spectra

    def download_region(self, target: str, radius_arcmin: float = 10,
                       mag_limit: float = 15, max_sources: int = 1000,
                       download_spectra: bool = True) -> Dict:
        """Download Gaia data for a named target."""
        ra, dec = self.resolve_target(target)
        print(f"Resolved '{target}' to RA={ra:.4f}°, Dec={dec:.4f}°")

        return self.download_region_coords(
            ra=ra, dec=dec,
            radius_deg=float(radius_arcmin) / 60.0,
            mag_limit=float(mag_limit),
            max_sources=int(max_sources),
            download_spectra=bool(download_spectra)
        )

    def download_region_coords(self, ra: float, dec: float, radius_deg: float,
                               mag_limit: float = 15, max_sources: int = 1000,
                               download_spectra: bool = True) -> Dict:
        """Download Gaia data for a coordinate region."""
        sources = self.query_gaia_sources(ra, dec, radius_deg, mag_limit, max_sources)

        if not sources:
            print("No sources found")
            return {'sources_found': 0, 'spectra_downloaded': 0}

        print(f"Adding {len(sources)} sources to database...")
        for source in sources:
            self.db.add_source(source)

        spectra_downloaded = 0
        if download_spectra:
            if not HAS_GAIAXPY:
                print("GaiaXPy not installed; skipping spectra download.")
            else:
                source_ids = [s.source_id for s in sources]
                spectra = self.download_xp_spectra(source_ids)

                print(f"Storing {len(spectra)} spectra in database...")
                for spectrum in spectra:
                    self.db.add_spectrum(spectrum)
                    spectra_downloaded += 1

        stats = {
            'sources_found': len(sources),
            'spectra_downloaded': spectra_downloaded,
            'center': (float(ra), float(dec)),
            'radius_deg': float(radius_deg)
        }

        print(f"\nDownload complete!")
        print(f"  Sources: {stats['sources_found']}")
        print(f"  Spectra: {stats['spectra_downloaded']}")
        print(f"  Database: {self.db}")

        return stats

    def close(self):
        """Close the database."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_download(target: str = "Vega", radius_arcmin: float = 5,
                  mag_limit: float = 12, db_path: str = "test_gaia.db"):
    """Test the downloader on a small region."""
    print(f"Testing Gaia downloader on '{target}'")
    print(f"Radius: {radius_arcmin} arcmin, Mag limit: {mag_limit}")
    print(f"Output: {db_path}")
    print("-" * 50)

    deps = []
    if not HAS_ASTROQUERY:
        deps.append("astroquery")
    if not HAS_ASTROPY:
        deps.append("astropy")
    if not HAS_GAIAXPY:
        deps.append("gaiaxpy")

    if deps:
        print(f"Missing dependencies: {', '.join(deps)}")
        print(f"Install with: pip install {' '.join(deps)}")
        return

    try:
        with GaiaDownloader(db_path) as downloader:
            if ',' in target:
                ra, dec = map(float, target.split(','))
                stats = downloader.download_region_coords(
                    ra=ra, dec=dec,
                    radius_deg=float(radius_arcmin) / 60.0,
                    mag_limit=float(mag_limit),
                    max_sources=100
                )
            else:
                stats = downloader.download_region(
                    target=target,
                    radius_arcmin=float(radius_arcmin),
                    mag_limit=float(mag_limit),
                    max_sources=100
                )

            if stats['spectra_downloaded'] > 0:
                print("\nSample spectrum:")
                db = GaiaSpectraDB(db_path)

                cursor = db._conn.cursor()
                cursor.execute('SELECT source_id FROM spectra LIMIT 1')
                row = cursor.fetchone()

                if row:
                    source_id = int(row[0])
                    spectrum = db.get_spectrum(source_id)
                    source = db.get_source(source_id)

                    print(f"  Source ID: {source_id}")
                    if source:
                        print(f"  G mag: {source.phot_g_mean_mag:.2f}" if source.phot_g_mean_mag is not None else "  G mag: N/A")
                        print(f"  BP-RP: {source.bp_rp:.3f}" if source.bp_rp is not None else "  BP-RP: N/A")
                    if spectrum:
                        print(f"  Wavelength range: {spectrum.wavelengths[0]:.0f} - {spectrum.wavelengths[-1]:.0f} nm")
                        print(f"  Peak flux at: {spectrum.wavelengths[int(np.argmax(spectrum.flux))]:.0f} nm")
                        print(f"  Flux at 500nm: {spectrum.get_flux_at(500):.3e}")
                        print(f"  Flux at 700nm: {spectrum.get_flux_at(700):.3e}")

                db.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "Vega"
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    mag_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 12

    test_download(target, radius, mag_limit)
