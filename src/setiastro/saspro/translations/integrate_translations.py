# -*- coding: utf-8 -*-
"""
Translation integration script for Seti Astro Suite Pro.

Usage:
    python integrate_translations.py

This will:
1. Load translations from it_translations.py, fr_translations.py, es_translations.py, zh_translations.py
2. Update the .ts files with translations
3. Compile .qm files using lrelease
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Get the translations directory
SCRIPT_DIR = Path(__file__).parent

# Import merged translation files
try:
    from it_translations import TRANSLATIONS_IT
except ImportError:
    print("Warning: Could not import Italian translations (it_translations.py)")
    TRANSLATIONS_IT = {}

try:
    from fr_translations import TRANSLATIONS_FR
except ImportError:
    print("Warning: Could not import French translations (fr_translations.py)")
    TRANSLATIONS_FR = {}

try:
    from es_translations import TRANSLATIONS_ES
except ImportError:
    print("Warning: Could not import Spanish translations (es_translations.py)")
    TRANSLATIONS_ES = {}

try:
    from zh_translations import TRANSLATIONS_ZH
except ImportError:
    print("Warning: Could not import Chinese translations (zh_translations.py)")
    TRANSLATIONS_ZH = {}

try:
    from de_translations import TRANSLATIONS_DE
except ImportError:
    print("Warning: Could not import German translations (de_translations.py)")
    TRANSLATIONS_DE = {}

try:
    from pt_translations import TRANSLATIONS_PT
except ImportError:
    print("Warning: Could not import Portuguese translations (pt_translations.py)")
    TRANSLATIONS_PT = {}

try:
    from ja_translations import TRANSLATIONS_JA
except ImportError:
    print("Warning: Could not import Japanese translations (ja_translations.py)")
    TRANSLATIONS_JA = {}


def inject_translations_into_ts(ts_file: Path, translations: dict, lang: str):
    """
    Inject translations into a .ts file.
    
    Args:
        ts_file: Path to the .ts file
        translations: Dict of {context: {source: translation}}
        lang: Language code (it, fr, es)
    """
    if not ts_file.exists():
        # Create a basic .ts file if it doesn't exist
        with open(ts_file, 'w', encoding='utf-8') as f:
            f.write(f'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n<TS version="2.1" language="{lang}">\n</TS>')
    
    # Parse the existing .ts file
    tree = ET.parse(ts_file)
    root = tree.getroot()
    
    injected_count = 0
    
    # Build a map of existing contexts for fast lookup
    context_elements = {}
    for context in root.findall('context'):
        name_elem = context.find('name')
        if name_elem is not None and name_elem.text:
            context_elements[name_elem.text] = context
    
    # Iterate through all contexts in our translations
    for context_name, context_translations in translations.items():
        # Find or create the context element
        if context_name in context_elements:
            context = context_elements[context_name]
        else:
            context = ET.SubElement(root, 'context')
            name_elem = ET.SubElement(context, 'name')
            name_elem.text = context_name
            context_elements[context_name] = context
        
        # Build map of existing source strings in this context
        existing_messages = {}
        for message in context.findall('message'):
            source = message.findtext('source')
            if source:
                existing_messages[source] = message
        
        # Iterate through all translations in this context
        for source, translation_text in context_translations.items():
            if source in existing_messages:
                message = existing_messages[source]
                # Find or create the translation element
                translation = message.find('translation')
                if translation is None:
                    translation = ET.SubElement(message, 'translation')
            else:
                # Create a new message element
                message = ET.SubElement(context, 'message')
                source_elem = ET.SubElement(message, 'source')
                source_elem.text = source
                translation = ET.SubElement(message, 'translation')
            
            # Set the translation text and remove 'unfinished' type
            translation.text = translation_text
            if 'type' in translation.attrib:
                del translation.attrib['type']
            
            injected_count += 1
    
    # Write back the modified .ts file with pretty printing if possible
    # We'll just use the default write for now
    tree.write(ts_file, encoding='utf-8', xml_declaration=True)
    
    return injected_count


def run_lrelease(ts_file: Path):
    """
    Compile a .ts file to .qm using pyside6-lrelease or lrelease.
    """
    import subprocess
    import sys
    
    qm_file = ts_file.with_suffix('.qm')
    
    # Try common command names
    cmds = ['pyside6-lrelease', 'lrelease']
    
    # Try to find lrelease in the same directory as sys.executable (often in Scripts on Windows)
    scripts_dir = Path(sys.executable).parent
    cmds.append(str(scripts_dir / 'pyside6-lrelease.exe'))
    cmds.append(str(scripts_dir / 'lrelease.exe'))
    
    # Try to find it in site-packages/PySide6 (where I found it)
    try:
        import PySide6
        pyside6_dir = Path(PySide6.__file__).parent
        cmds.append(str(pyside6_dir / 'lrelease.exe'))
        cmds.append(str(pyside6_dir / 'pyside6-lrelease.exe'))
    except ImportError:
        pass

    for cmd in cmds:
        try:
            result = subprocess.run(
                [cmd, str(ts_file), '-qm', str(qm_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"    Success using {cmd}")
                return True
        except FileNotFoundError:
            continue
    
    return False


def main():
    print("=" * 60)
    print("Seti Astro Suite Pro - Translation Integration")
    print("=" * 60)
    
    # Translation data
    translations_data = {
        'it': TRANSLATIONS_IT,
        'fr': TRANSLATIONS_FR,
        'es': TRANSLATIONS_ES,
        'zh': TRANSLATIONS_ZH,
        'de': TRANSLATIONS_DE,
        'pt': TRANSLATIONS_PT,
        'ja': TRANSLATIONS_JA,
    }
    
    # Print summary
    print("\n[1] Loading translations...")
    for lang, data in translations_data.items():
        string_count = sum(len(v) for v in data.values())
        context_count = len(data)
        print(f"  {lang.upper()}: {string_count} strings in {context_count} contexts")
    
    # Inject translations into .ts files
    print("\n[2] Injecting translations into .ts files...")
    for lang, data in translations_data.items():
        ts_file = SCRIPT_DIR / f"saspro_{lang}.ts"
        count = inject_translations_into_ts(ts_file, data, lang)
        print(f"  {lang.upper()}: Injected {count} translations into {ts_file.name}")
    
    # Compile .qm files
    print("\n[3] Compiling .qm files...")
    for lang in translations_data.keys():
        ts_file = SCRIPT_DIR / f"saspro_{lang}.ts"
        if ts_file.exists():
            if run_lrelease(ts_file):
                print(f"  Compiled {ts_file.name} -> {ts_file.stem}.qm")
            else:
                print(f"  Warning: Could not compile {ts_file.name}")
    
    print("\n" + "=" * 60)
    print("Translation integration complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Restart the application")
    print("2. Go to Settings > Language")
    print("3. Select your preferred language")
    print("4. Restart to apply changes")


if __name__ == '__main__':
    main()
