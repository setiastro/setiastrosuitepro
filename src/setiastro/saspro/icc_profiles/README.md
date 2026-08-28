# ICC Profiles

SASpro checks this directory for optional export/display ICC profiles.

Recognized filename variants include:

- `DisplayP3.icc`, `Display P3.icc`
- `Adobe RGB (1998).icc`, `AdobeRGB1998.icc`
- `ProPhoto.icc`, `ProPhoto RGB.icc`, `ROMM RGB.icc`
- `sRGB.icc`

sRGB can also be generated at runtime through Pillow, so a bundled sRGB profile
is optional.

Adobe RGB (1998) profiles are subject to Adobe's ICC Profile Bundling Agreement.
Do not bundle Adobe profiles unless the distribution complies with that agreement.