from pathlib import Path
import sqlite3
import tempfile
import unittest
from agent import Tools, LocalPlanner, calculate, run


class AgentTests(unittest.TestCase):
    def test_tools_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"guide.txt").write_text("Deployment checklist: run tests first.")
            tools=Tools(root)
            self.assertEqual(calculate("(12+8)/4"),5)
            self.assertIn("guide.txt",tools.run("search_docs",{"query":"deployment"}))
            with self.assertRaises(ValueError): tools.run("read_file",{"path":"../outside.txt"})
            with sqlite3.connect(":memory:") as db:
                run_id,answer=run("find: deployment",LocalPlanner(),tools,db)
                self.assertIn("guide.txt",answer)
                self.assertEqual(db.execute("SELECT count(*) FROM steps WHERE run_id=?",(run_id,)).fetchone()[0],2)

    def test_bad_call_reported_and_limit(self):
        class Planner:
            def decide(self,question,messages):
                return {"tool":"calculator","arguments":{"expression":"__import__('os')"}}
        with tempfile.TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
            run_id,answer=run("unsafe",Planner(),Tools(Path(directory)),db,max_steps=3)
            self.assertEqual(answer,"Step limit reached")
            self.assertEqual(db.execute("SELECT count(*) FROM steps WHERE run_id=? AND result LIKE 'Tool error:%'",(run_id,)).fetchone()[0],3)


if __name__ == "__main__": unittest.main()

