from typing import Optional, List

from pydantic import BaseModel, Field


class Path(BaseModel):
    current: str
    tail: Optional[List["Path"]] = Field(default=None)

    def as_str_list(self) -> List[str]:
        """
        >>> Path(current='a', tail=[Path(current='b', tail=[Path(current='c'), Path(current='d')])]).as_str_list()
        ['a.b.c', 'a.b.d']
        """

        # Recursive function to collect all paths
        def collect_paths(path: Path, prefix: str = "") -> List[str]:
            current_path = prefix + path.current
            if not path.tail:
                return [current_path]
            else:
                paths = []
                for sub_path in path.tail:
                    paths.extend(collect_paths(sub_path, current_path + "."))
                return paths

        # Collect all paths starting from this object
        return collect_paths(self)


def convert_paths(paths: List[str]) -> List[Path]:
    """Convert string paths into Path objects

    Paths which share the same root are grouped together.

    Args:
        paths: List[str]: List of str paths containing "." as separator

    Returns:
        List[Path]: List of Path objects
    """
    sorted_paths = sorted(paths)
    prev_root = None
    converted_paths = []
    for path in sorted_paths:
        parts = path.split(".")
        root = parts[0]
        if root != prev_root:
            converted_paths.append(Path(current=root))
            prev_root = root
        current = converted_paths[-1]
        for part in parts[1:]:
            if current.tail is None:
                current.tail = []
            found = False
            for tail in current.tail:
                if tail.current == part:
                    current = tail
                    found = True
                    break
            if not found:
                new_tail = Path(current=part)
                assert current.tail is not None
                current.tail.append(new_tail)
                current = new_tail
    return converted_paths