from unittest.mock import MagicMock
from uuid import UUID
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.document_intelligence.evidence import AnalysisBudget, DocumentEvidenceService
from app.documents.repository import DocumentRepository

WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
DOCUMENT_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")
VERSION_ID = UUID("8f3bcfc5-7c76-4196-8dad-214df37f2121")


def test_analysis_chunk_page_is_workspace_document_version_and_member_scoped() -> None:
    session = MagicMock(spec=Session)
    session.execute.return_value.all.return_value = []
    repository = DocumentRepository(session)

    assert repository.list_version_analysis_chunk_page(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
        user_id=USER_ID,
        after_chunk_index=4,
        limit=25,
    ) == []

    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    params = compiled.params
    assert "documents.workspace_id =" in sql
    assert "documents.id =" in sql
    assert "document_versions.id =" in sql
    assert "document_versions.document_id =" in sql
    assert "workspace_members.workspace_id =" in sql
    assert "workspace_members.user_id =" in sql
    assert "document_chunks.chunk_index >" in sql
    assert "ORDER BY document_chunks.chunk_index ASC" in sql
    selected_columns = sql.split("FROM document_chunks", 1)[0]
    assert "document_chunks.embedding" not in selected_columns
    assert WORKSPACE_ID in params.values()
    assert DOCUMENT_ID in params.values()
    assert VERSION_ID in params.values()
    assert USER_ID in params.values()
    assert 4 in params.values()
    assert 25 in params.values()
    session.commit.assert_called_once()


def test_analysis_chunk_page_rejects_nonpositive_limit() -> None:
    repository = DocumentRepository(MagicMock(spec=Session))

    try:
        repository.list_version_analysis_chunk_page(
            workspace_id=WORKSPACE_ID,
            document_id=DOCUMENT_ID,
            version_id=VERSION_ID,
            user_id=USER_ID,
            after_chunk_index=None,
            limit=0,
        )
    except ValueError as error:
        assert "greater than zero" in str(error)
    else:
        raise AssertionError("nonpositive limit was not rejected")


class EvidenceRepositoryFake:
    def __init__(self, *, workspace_id: UUID = WORKSPACE_ID, can_access_version: bool = True) -> None:
        self.document = SimpleNamespace(
            id=DOCUMENT_ID,
            workspace_id=workspace_id,
            filename="evidence.pdf",
            processing_status="ready",
            versions=[
                SimpleNamespace(
                    id=VERSION_ID,
                    document_id=DOCUMENT_ID,
                    version_number=7,
                    processing_status="ready",
                )
            ],
        )
        self.can_access_version = can_access_version
        self.events: list[str] = []

    def get_for_user(self, document_id: UUID, user_id: UUID):
        self.events.append("document_access")
        return self.document if document_id == DOCUMENT_ID else None

    def user_can_access_version(self, document_id: UUID, version_id: UUID, user_id: UUID) -> bool:
        self.events.append("version_access")
        return self.can_access_version and document_id == DOCUMENT_ID and version_id == VERSION_ID

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")

    def list_version_analysis_chunk_page(self, **kwargs):
        self.events.append("chunk_page")
        chunk = SimpleNamespace(
            id=UUID("08be4613-06af-44d9-b2c4-7ff864c9eb41"),
            chunk_index=0,
            text="Authorized evidence text",
            estimated_token_count=5,
            source_metadata={"page_numbers": [1]},
        )
        return [(chunk, self.document, self.document.versions[0])]


def test_evidence_service_checks_document_and_version_before_loading_chunks() -> None:
    repository = EvidenceRepositoryFake()
    evidence_service = DocumentEvidenceService(repository)  # type: ignore[arg-type]

    metadata, chunks = evidence_service.load_version(
        WORKSPACE_ID,
        DOCUMENT_ID,
        VERSION_ID,
        str(USER_ID),
        AnalysisBudget(),
    )

    assert metadata.version_number == 7
    assert chunks[0].source_id == "S1"
    assert repository.events == [
        "document_access",
        "version_access",
        "commit",
        "chunk_page",
    ]


@pytest.mark.parametrize(
    ("repo", "requested_workspace", "requested_version"),
    [
        (EvidenceRepositoryFake(workspace_id=UUID("22119a02-61eb-4430-94af-1f8ec1078c7b")), WORKSPACE_ID, VERSION_ID),
        (EvidenceRepositoryFake(can_access_version=False), WORKSPACE_ID, VERSION_ID),
        (EvidenceRepositoryFake(), WORKSPACE_ID, UUID("cd5b0042-98da-448d-b9ae-55d4558d0b8a")),
    ],
)
def test_evidence_service_rejects_cross_workspace_or_inaccessible_versions(
    repo: EvidenceRepositoryFake,
    requested_workspace: UUID,
    requested_version: UUID,
) -> None:
    evidence_service = DocumentEvidenceService(repo)  # type: ignore[arg-type]

    with pytest.raises(AppError) as error:
        evidence_service.load_version(
            requested_workspace,
            DOCUMENT_ID,
            requested_version,
            str(USER_ID),
            AnalysisBudget(),
        )

    assert error.value.status_code == 404
    assert "chunk_page" not in repo.events