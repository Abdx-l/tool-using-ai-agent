# Tool-Using AI Agent

An explicit tool-call loop with a ten-step limit, argument validation, and SQLite trace logs. It provides a document search tool, a safe arithmetic calculator, and a file reader restricted to the configured document directory. Tool errors are returned to the model as messages so it can retry.

```sh
python agent.py --docs examples/docs --db runs.db "find: deployment"
python agent.py --docs examples/docs --db runs.db "calc: (12 + 8) / 4"
python agent.py --replay 1 --db runs.db
python -m unittest discover -s tests -v
```

The default local planner supports the examples without an account. To use a chat completions model with function calling, set `OPENAI_API_KEY` and pass `--endpoint URL --model MODEL`. The request and tool results are stored in the local trace database.

