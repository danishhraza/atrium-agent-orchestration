import os
import tempfile

# Must run before `app` is imported: isolated database, dev auth, offline embeddings, no tracing.
os.environ["ATRIUM_DATA_DIR"] = tempfile.mkdtemp(prefix="atrium-test-")
os.environ["ATRIUM_DEV_AUTH"] = "1"
os.environ["ATRIUM_EMBEDDING_MODEL"] = "local"
os.environ["ATRIUM_DATABASE_URL"] = ""
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)

import pytest  # noqa: E402

from app import llm  # noqa: E402
from app.db import init_db  # noqa: E402
from tests import fake_llm  # noqa: E402

init_db()


@pytest.fixture(autouse=True)
def scripted_llm(monkeypatch):
    """No test ever calls a real model."""
    monkeypatch.setattr(llm, "complete", fake_llm.complete)
