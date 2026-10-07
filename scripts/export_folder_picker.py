"""Start destination selection beside the previous folder, rather than inside it."""
import json
from pathlib import Path


class ExportFolderPicker:
    def __init__(self, choose, settings_path, fallback=None):
        self.choose = choose
        self.settings_path = Path(settings_path)
        self.fallback = Path(fallback or Path.home() / 'Documents')
        try:
            self.last = json.loads(self.settings_path.read_text(encoding='utf-8')).get('directory', '')
        except (OSError, ValueError, AttributeError):
            self.last = ''
        if not isinstance(self.last, str):
            self.last = ''

    def __call__(self, current_directory=''):
        current = current_directory if isinstance(current_directory, str) else ''
        previous = Path(current.strip() or self.last).expanduser() if current.strip() or self.last else None
        initial = previous.parent if previous and previous.is_absolute() else self.fallback
        while not initial.is_dir() and initial.parent != initial:
            initial = initial.parent
        result = self.choose(str(initial))
        if not result:
            return ''
        self.last = str(Path(result).expanduser())
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            pending = self.settings_path.with_suffix('.tmp')
            pending.write_text(json.dumps(dict(directory=self.last)), encoding='utf-8')
            pending.replace(self.settings_path)
        except OSError:
            pass
        return self.last
