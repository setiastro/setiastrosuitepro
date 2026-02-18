# gaia_downloader.py
# Downloads Gaia DR3 XP spectra and creates our own local database

from __future__ import annotations

import os
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np

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
    phot_g_mean_mag: float
    bp_rp: Optional[float] = None
    parallax: Optional[float] = None
    pmra: Optional[float] = None
    pmdec: Optional[float] = None


@dataclass
class CalibratedSpectrum:
    """A calibrated XP spectrum."""
    source_id: int
    wavelengths: np.ndarray  # nm
    flux: np.ndarray  # W/nm/m^2
    flux_error: Optional[np.ndarray] = None

    @property
    def normalized_flux(self) -> np.ndarray:
        """Flux normalized to 0-1 range."""
        if self.flux.max() > 0:
            return self.flux / self.flux.max()
        return self.flux

    def get_flux_at(self, wavelength_nm: float) -> float:
        """Interpolate flux at specific wavelength."""
        return float(np.interp(wavelength_nm, self.wavelengths, self.flux))


class GaiaSpectraDB:
    """
    Local database for storing Gaia XP spectra.

    Uses SQLite for metadata and compressed numpy arrays for spectra.
    This is a simple format optimized for our SFCC use case.
    """

    # Standard wavelength grid (matches GaiaXPy default output)
    WAVELENGTH_START = 336  # nm
    WAVELENGTH_END = 1020   # nm
    WAVELENGTH_STEP = 2     # nm
    N_WAVELENGTHS = 343

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
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

        # Create spatial index
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
        """Standard wavelength grid."""
        return np.linspace(self.WAVELENGTH_START, self.WAVELENGTH_END,
                          self.N_WAVELENGTHS)

    def _compress_array(self, arr: np.ndarray) -> bytes:
        """Compress numpy array to bytes."""
        # Convert to float32 for compact storage
        arr_f32 = arr.astype(np.float32)
        return zlib.compress(arr_f32.tobytes(), level=6)

    def _decompress_array(self, data: bytes) -> np.ndarray:
        """Decompress bytes to numpy array."""
        decompressed = zlib.decompress(data)
        return np.frombuffer(decompressed, dtype=np.float32)

    def add_source(self, source: GaiaSource):
        """Add a source to the database."""
        cursor = self._conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sources
            (source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (source.source_id, source.ra, source.dec, source.phot_g_mean_mag,
              source.bp_rp, source.parallax, source.pmra, source.pmdec))
        self._conn.commit()

    def add_spectrum(self, spectrum: CalibratedSpectrum):
        """Add a calibrated spectrum to the database."""
        cursor = self._conn.cursor()

        # Resample to standard grid if needed
        if len(spectrum.wavelengths) != self.N_WAVELENGTHS:
            flux_resampled = np.interp(self.wavelengths, spectrum.wavelengths,
                                       spectrum.flux, left=0, right=0)
            if spectrum.flux_error is not None:
                error_resampled = np.interp(self.wavelengths, spectrum.wavelengths,
                                           spectrum.flux_error, left=0, right=0)
            else:
                error_resampled = None
        else:
            flux_resampled = spectrum.flux
            error_resampled = spectrum.flux_error

        # Compress and store
        flux_compressed = self._compress_array(flux_resampled)
        error_compressed = self._compress_array(error_resampled) if error_resampled is not None else None

        cursor.execute('''
            INSERT OR REPLACE INTO spectra (source_id, flux_compressed, flux_error_compressed)
            VALUES (?, ?, ?)
        ''', (spectrum.source_id, flux_compressed, error_compressed))

        # Update source flag
        cursor.execute('''
            UPDATE sources SET has_xp_spectrum = 1 WHERE source_id = ?
        ''', (spectrum.source_id,))

        self._conn.commit()

    def get_spectrum(self, source_id: int) -> Optional[CalibratedSpectrum]:
        """Retrieve a spectrum by source ID."""
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT flux_compressed, flux_error_compressed FROM spectra
            WHERE source_id = ?
        ''', (source_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        flux = self._decompress_array(row[0])
        flux_error = self._decompress_array(row[1]) if row[1] else None

        return CalibratedSpectrum(
            source_id=source_id,
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
        ''', (source_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return GaiaSource(*row)

    def query_region(self, ra: float, dec: float, radius_deg: float,
                    mag_limit: Optional[float] = None,
                    only_with_spectra: bool = False) -> List[GaiaSource]:
        """Query sources in a circular region."""
        cursor = self._conn.cursor()

        # Simple box query (for small regions)
        ra_min = ra - radius_deg / np.cos(np.radians(dec))
        ra_max = ra + radius_deg / np.cos(np.radians(dec))
        dec_min = dec - radius_deg
        dec_max = dec + radius_deg

        query = '''
            SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec
            FROM sources
            WHERE ra BETWEEN ? AND ?
            AND dec BETWEEN ? AND ?
        '''
        params = [ra_min, ra_max, dec_min, dec_max]

        if mag_limit:
            query += ' AND phot_g_mean_mag <= ?'
            params.append(mag_limit)

        if only_with_spectra:
            query += ' AND has_xp_spectrum = 1'

        cursor.execute(query, params)

        sources = []
        for row in cursor.fetchall():
            # Verify actual distance
            d = np.sqrt((row[1] - ra)**2 * np.cos(np.radians(dec))**2 +
                       (row[2] - dec)**2)
            if d <= radius_deg:
                sources.append(GaiaSource(*row))

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

        # Download stars around a target
        downloader.download_region("M31", radius_arcmin=10, mag_limit=15)

        # Or by coordinates
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
        """Resolve target name to coordinates using SIMBAD."""
        try:
            coord = SkyCoord.from_name(target_name)
            return coord.ra.deg, coord.dec.deg
        except Exception as e:
            raise ValueError(f"Could not resolve target '{target_name}': {e}")

    def query_gaia_sources(self, ra: float, dec: float, radius_deg: float,
                           mag_limit: float = 17.0,
                           max_sources: int = 10000) -> List[GaiaSource]:
        """
        Query Gaia DR3 for sources in a region.

        Args:
            ra, dec: Center coordinates in degrees
            radius_deg: Search radius in degrees
            mag_limit: Maximum G magnitude
            max_sources: Maximum number of sources to return

        Returns:
            List of GaiaSource objects
        """
        # ADQL query for Gaia DR3
        query = f"""
        SELECT TOP {max_sources}
            source_id,
            ra,
            dec,
            phot_g_mean_mag,
            bp_rp,
            parallax,
            pmra,
            pmdec,
            has_xp_continuous
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        )
        AND phot_g_mean_mag < {mag_limit}
        AND has_xp_continuous = 'True'
        ORDER BY phot_g_mean_mag ASC
        """

        print(f"Querying Gaia DR3 for sources within {radius_deg}° of ({ra:.4f}, {dec:.4f})...")

        job = Gaia.launch_job_async(query)
        result = job.get_results()

        sources = []
        for row in result:
            sources.append(GaiaSource(
                source_id=int(row['source_id']),
                ra=float(row['ra']),
                dec=float(row['dec']),
                phot_g_mean_mag=float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else None,
                bp_rp=float(row['bp_rp']) if row['bp_rp'] else None,
                parallax=float(row['parallax']) if row['parallax'] else None,
                pmra=float(row['pmra']) if row['pmra'] else None,
                pmdec=float(row['pmdec']) if row['pmdec'] else None
            ))

        print(f"Found {len(sources)} sources with XP spectra")
        return sources

    def download_xp_spectra(self, source_ids: List[int],
                            batch_size: int = 100) -> List[CalibratedSpectrum]:
        """
        Download and calibrate XP spectra for given source IDs.

        Args:
            source_ids: List of Gaia DR3 source IDs
            batch_size: Number of sources per batch

        Returns:
            List of CalibratedSpectrum objects
        """
        if not HAS_GAIAXPY:
            raise ImportError("GaiaXPy is required for spectrum calibration")

        all_spectra = []

        for i in range(0, len(source_ids), batch_size):
            batch_ids = source_ids[i:i + batch_size]
            print(f"Downloading spectra {i+1}-{min(i+batch_size, len(source_ids))} of {len(source_ids)}...")

            try:
                # GaiaXPy calibrate function returns (dataframe, sampling)
                # sampling contains the wavelength grid
                calibrated, sampling = calibrate(batch_ids)

                # Get wavelength array from sampling (or use default grid)
                if sampling is not None and hasattr(sampling, '__iter__'):
                    wavelengths = np.array(sampling)
                else:
                    # Default GaiaXPy wavelength grid: 336-1020nm, 343 samples
                    wavelengths = np.linspace(336, 1020, 343)

                for _, row in calibrated.iterrows():
                    flux = np.array(row['flux'])
                    flux_error = np.array(row['flux_error']) if 'flux_error' in calibrated.columns else None

                    spectrum = CalibratedSpectrum(
                        source_id=int(row['source_id']),
                        wavelengths=wavelengths,
                        flux=flux,
                        flux_error=flux_error
                    )
                    all_spectra.append(spectrum)

            except Exception as e:
                print(f"Warning: Failed to download batch: {e}")
                continue

        return all_spectra

    def download_region(self, target: str, radius_arcmin: float = 10,
                       mag_limit: float = 15, max_sources: int = 1000,
                       download_spectra: bool = True) -> Dict:
        """
        Download Gaia data for a named target.

        Args:
            target: Target name (e.g., "M31", "NGC 7000", "Vega")
            radius_arcmin: Search radius in arcminutes
            mag_limit: Maximum G magnitude
            max_sources: Maximum sources to download
            download_spectra: Whether to download XP spectra

        Returns:
            Dictionary with download statistics
        """
        ra, dec = self.resolve_target(target)
        print(f"Resolved '{target}' to RA={ra:.4f}°, Dec={dec:.4f}°")

        return self.download_region_coords(
            ra=ra, dec=dec,
            radius_deg=radius_arcmin / 60,
            mag_limit=mag_limit,
            max_sources=max_sources,
            download_spectra=download_spectra
        )

    def download_region_coords(self, ra: float, dec: float, radius_deg: float,
                               mag_limit: float = 15, max_sources: int = 1000,
                               download_spectra: bool = True) -> Dict:
        """
        Download Gaia data for a coordinate region.

        Args:
            ra, dec: Center coordinates in degrees
            radius_deg: Search radius in degrees
            mag_limit: Maximum G magnitude
            max_sources: Maximum sources to download
            download_spectra: Whether to download XP spectra

        Returns:
            Dictionary with download statistics
        """
        # Query sources
        sources = self.query_gaia_sources(ra, dec, radius_deg, mag_limit, max_sources)

        if not sources:
            print("No sources found")
            return {'sources_found': 0, 'spectra_downloaded': 0}

        # Add sources to database
        print(f"Adding {len(sources)} sources to database...")
        for source in sources:
            self.db.add_source(source)

        # Download spectra if requested
        spectra_downloaded = 0
        if download_spectra and HAS_GAIAXPY:
            source_ids = [s.source_id for s in sources]
            spectra = self.download_xp_spectra(source_ids)

            print(f"Storing {len(spectra)} spectra in database...")
            for spectrum in spectra:
                self.db.add_spectrum(spectrum)
                spectra_downloaded += 1

        stats = {
            'sources_found': len(sources),
            'spectra_downloaded': spectra_downloaded,
            'center': (ra, dec),
            'radius_deg': radius_deg
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
    """
    Test the downloader on a small region.

    Args:
        target: Target name or "ra,dec" coordinates
        radius_arcmin: Search radius in arcminutes
        mag_limit: Maximum magnitude
        db_path: Output database path
    """
    print(f"Testing Gaia downloader on '{target}'")
    print(f"Radius: {radius_arcmin} arcmin, Mag limit: {mag_limit}")
    print(f"Output: {db_path}")
    print("-" * 50)

    # Check dependencies
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
            # Handle coordinate input
            if ',' in target:
                ra, dec = map(float, target.split(','))
                stats = downloader.download_region_coords(
                    ra=ra, dec=dec,
                    radius_deg=radius_arcmin / 60,
                    mag_limit=mag_limit,
                    max_sources=100
                )
            else:
                stats = downloader.download_region(
                    target=target,
                    radius_arcmin=radius_arcmin,
                    mag_limit=mag_limit,
                    max_sources=100
                )

            # Show sample spectrum
            if stats['spectra_downloaded'] > 0:
                print("\nSample spectrum:")
                db = GaiaSpectraDB(db_path)

                # Get first source with spectrum
                cursor = db._conn.cursor()
                cursor.execute('SELECT source_id FROM spectra LIMIT 1')
                row = cursor.fetchone()

                if row:
                    source_id = row[0]
                    spectrum = db.get_spectrum(source_id)
                    source = db.get_source(source_id)

                    print(f"  Source ID: {source_id}")
                    if source:
                        print(f"  G mag: {source.phot_g_mean_mag:.2f}")
                        print(f"  BP-RP: {source.bp_rp:.3f}" if source.bp_rp else "  BP-RP: N/A")
                    print(f"  Wavelength range: {spectrum.wavelengths[0]:.0f} - {spectrum.wavelengths[-1]:.0f} nm")
                    print(f"  Peak flux at: {spectrum.wavelengths[np.argmax(spectrum.flux)]:.0f} nm")
                    print(f"  Flux at 500nm: {spectrum.get_flux_at(500):.3e}")
                    print(f"  Flux at 700nm: {spectrum.get_flux_at(700):.3e}")

                db.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "Vega"  # Bright star, should have XP spectrum

    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 5  # arcmin
    mag_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 12

    test_download(target, radius, mag_limit)
