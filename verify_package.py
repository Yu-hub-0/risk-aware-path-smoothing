"""Check portable links and packaged original-file integrity without simulation."""
from pathlib import Path
import hashlib,json,re

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'snapshot_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        p=ROOT/name
        assert p.is_file(),('Missing file',name)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,('Snapshot changed',name)
    for md in ROOT.rglob('README.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):
                assert (md.parent/target.split('#')[0]).exists(),('Broken link',target)
    print('PASS: original experiment snapshot hashes and README links')

if __name__=='__main__':main()
