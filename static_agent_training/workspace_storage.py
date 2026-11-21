"""Validated, reversible removal of workspace sessions and exported populations."""
import json
from pathlib import Path
import uuid

from .catalog import catalog, read_json
from .status import training_processes


class WorkspaceStorage:
    def __init__(self, workspace):
        self.workspace = workspace
        self.root = workspace.root
        self.sessions = workspace.sessions

    def active(self):
        records = [r for r in self.sessions.records() if r['state'] in ('running','stopping')]
        paths = {Path(p).resolve() for p in training_processes()}
        paths.update(Path(r['directory']).resolve() for r in records)
        # Also protect unmanaged assessment/comparison processes, not just trainers.
        for proc in Path('/proc').iterdir():
            try:
                args = (proc/'cmdline').read_bytes().decode().split('\0')
                if not any(a.startswith('static_agent_training.') for a in args):continue
                for flag in ('--output','--resume'):
                    if flag in args:
                        raw=Path(args[args.index(flag)+1])
                        paths.add((raw if raw.is_absolute() else (proc/'cwd').resolve()/raw).resolve())
            except (OSError,ValueError,IndexError,UnicodeError):continue
        return records, paths

    def protect(self, target, relative):
        records, paths = self.active()
        for path in paths:
            if path == target or path.is_relative_to(target) or target.is_relative_to(path):
                raise ValueError('Stop the active session before deleting this item')
        # Check explicit references in active session configurations, including source
        # populations used during startup and source structures used by adaptation.
        documents = [r.get('config',{}) for r in records]
        for path in paths:
            documents.extend(read_json(path/name) for name in ('config.json','plan.json','manifest.json'))
        def references(value):
            if isinstance(value,dict):return any(references(v) for v in value.values())
            if isinstance(value,list):return any(references(v) for v in value)
            if not isinstance(value,str):return False
            if value == relative:return True
            candidate=Path(value)
            if not candidate.is_absolute():return False
            resolved=candidate.resolve()
            return resolved==target or resolved.is_relative_to(target)
        if any(references(d) for d in documents):
            raise ValueError('An active session references this item; stop it before deletion')

    def delete(self, kind, relative):
        # Share the launch/resume lock, so managed jobs cannot start between checking
        # references and moving their input. The workspace lock serializes deletions.
        with self.workspace.lock, self.sessions.lock:
            if not isinstance(relative,str) or '..' in Path(relative).parts:
                raise ValueError('Choose a saved workspace item')
            base=self.root/'runs'
            lexical=base/relative
            if kind=='population':
                target=self.sessions.population_path(relative)
            elif kind=='session':
                target=self.sessions.path(relative)
                known={Path(r['path']).resolve() for r in catalog(self.root)}
                known.update(Path(r['directory']).resolve() for r in self.sessions.records())
                if target not in known:raise ValueError('Choose a saved session, not a container directory')
            else:raise ValueError('Unknown deletion type')
            if lexical.absolute()!=target or lexical.is_symlink():
                raise ValueError('Cannot delete workspace items through symbolic links')
            self.protect(target,relative)
            sources=[target]
            if kind=='session':
                sources += [target.parent/(target.name+'.log'),target.parent/(target.name+'-diagnostics')]
                sources += [p for p in self.sessions.registry.glob('*.json')
                            if read_json(p).get('directory')==str(target)]
            sources=[p for p in sources if p.exists() or p.is_symlink()]
            if any(p.is_symlink() or not p.resolve().is_relative_to(base.resolve()) for p in sources):
                raise ValueError('Unsafe linked session artifact')
            trash_root=self.root/'.workspace-trash'
            if trash_root.is_symlink():raise ValueError('Unsafe linked trash directory')
            trash=trash_root/uuid.uuid4().hex
            trash.mkdir(parents=True)
            moves=[(p,trash/str(i)/p.name) for i,p in enumerate(sources)]
            manifest=dict(kind=kind,item=relative,moves=[dict(source=str(p.relative_to(self.root)),stored=str(q.relative_to(trash))) for p,q in moves])
            (trash/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            moved=[]
            try:
                for source,dest in moves:
                    dest.parent.mkdir();source.rename(dest);moved.append((source,dest))
            except BaseException:
                for source,dest in reversed(moved):dest.rename(source)
                raise
            self.workspace.invalidate()
            return dict(deleted=relative,kind=kind,trash=str(trash.relative_to(self.root)))
