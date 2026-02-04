#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seti Astro Suite Pro - Main Entry Point

Backwards compatibility shim.
"""

from setiastro.saspro.__main__ import entry

if __name__ == "__main__":
    raise SystemExit(entry())
