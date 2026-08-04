import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RepositoryCheckoutContractTests(unittest.TestCase):
    def test_all_tracked_tsv_files_are_lf_only(self):
        attributes_path = REPO_ROOT / ".gitattributes"
        self.assertTrue(
            attributes_path.is_file(),
            "the repository must define a .gitattributes checkout policy",
        )

        tracked = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        tracked_tsv = [path for path in tracked if path.lower().endswith(".tsv")]
        self.assertTrue(tracked_tsv, "the LF policy test requires a tracked TSV fixture")

        for relative_path in tracked_tsv:
            with self.subTest(path=relative_path):
                eol = subprocess.run(
                    ["git", "check-attr", "eol", "--", relative_path],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.assertEqual(eol, f"{relative_path}: eol: lf")
                self.assertNotIn(
                    b"\r",
                    (REPO_ROOT / relative_path).read_bytes(),
                    f"{relative_path} must contain LF-only line endings",
                )


if __name__ == "__main__":
    unittest.main()
