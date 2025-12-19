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


def inject_translations_into_ts(ts_file: Path, translations: dict, lang: str):
    """
    Inject translations into a .ts file.
    
    Args:
        ts_file: Path to the .ts file
        translations: Dict of {context: {source: translation}}
        lang: Language code (it, fr, es)
    """
    if not ts_file.exists():
        print(f"Warning: {ts_file} does not exist")
        return 0
    
    # Parse the existing .ts file
    tree = ET.parse(ts_file)
    root = tree.getroot()
    
    injected_count = 0
    
    # Build a map of existing contexts for fast lookup
    context_elements = {}
    for context in root.findall('context'):
        context_name = context.findtext('name')
        if context_name:
            context_elements[context_name] = context
    
    # Iterate through all contexts in the .ts file
    for context in root.findall('context'):
        context_name = context.findtext('name')
        
        # Check if we have translations for this context
        if context_name not in translations:
            continue
        
        context_translations = translations[context_name]
        
        # Build set of existing source strings in this context
        existing_sources = set()
        for message in context.findall('message'):
            source = message.findtext('source')
            if source:
                existing_sources.add(source)
        
        # Iterate through all messages in this context
        for message in context.findall('message'):
            source = message.findtext('source')
            if not source:
                continue
            
            # Check if we have a translation for this source string
            if source not in context_translations:
                continue
            
            translation_text = context_translations[source]
            
            # Find or create the translation element
            translation = message.find('translation')
            if translation is None:
                translation = ET.SubElement(message, 'translation')
            
            # Set the translation text and remove 'unfinished' type
            translation.text = translation_text
            if 'type' in translation.attrib:
                del translation.attrib['type']
            
            injected_count += 1
        
        # Add new messages for translations that don't exist yet in this context
        for source, translation_text in context_translations.items():
            if source not in existing_sources:
                # Create a new message element
                new_message = ET.SubElement(context, 'message')
                source_elem = ET.SubElement(new_message, 'source')
                source_elem.text = source
                translation_elem = ET.SubElement(new_message, 'translation')
                translation_elem.text = translation_text
                injected_count += 1
    
    # Write back the modified .ts file
    tree.write(ts_file, encoding='utf-8', xml_declaration=True)
    
    return injected_count


def run_lrelease(ts_file: Path):
    """
    Compile a .ts file to .qm using pyside6-lrelease or lrelease.
    """
    import subprocess
    
    qm_file = ts_file.with_suffix('.qm')
    
    # Try pyside6-lrelease first
    for cmd in ['pyside6-lrelease', 'lrelease']:
        try:
            result = subprocess.run(
                [cmd, str(ts_file), '-qm', str(qm_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
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
        if ts_file.exists():
            count = inject_translations_into_ts(ts_file, data, lang)
            print(f"  {lang.upper()}: Injected {count} translations into {ts_file.name}")
        else:
            print(f"  {lang.upper()}: Warning - {ts_file.name} not found")
    
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
