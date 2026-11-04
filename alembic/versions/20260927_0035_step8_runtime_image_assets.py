"""Expose canonical images and routing search fields to online readers."""

from alembic import op

revision = "20260927_0035"
down_revision = "20260922_0034"
branch_labels = None
depends_on = None

_ONLINE_VIEW_ADDITIONS = (
    "runtime_image_assets_v1", "runtime_qbank_questions_v1", "runtime_qbank_paraphrases_v1",
)


def upgrade() -> None:
    op.execute("""CREATE VIEW pmmc_runtime.runtime_image_assets_v1
        WITH (security_barrier = true) AS
        SELECT a.snapshot_id, a.asset_id, a.local_path, a.remote_url,
               a.storage_uri, a.sha256, a.mime_type, a.width, a.height
          FROM pmmc_runtime.assets a
         WHERE a.asset_type IN ('raw_image', 'resized_image') AND EXISTS (
             SELECT 1 FROM pmmc_runtime.images i
              WHERE i.snapshot_id=a.snapshot_id AND i.asset_id=a.asset_id)
    """)
    op.execute("""CREATE VIEW pmmc_runtime.runtime_qbank_questions_v1
        WITH (security_barrier = true) AS
        SELECT q.qbank_id, q.canonical_question, q.search_document
          FROM pmmc_runtime.qbank_questions q
         WHERE EXISTS (SELECT 1 FROM pmmc_runtime.verified_qbank_routing_v4 r
                        WHERE r.qbank_id=q.qbank_id AND r.compiler_run_id=q.run_id)
    """)
    op.execute("""CREATE VIEW pmmc_runtime.runtime_qbank_paraphrases_v1
        WITH (security_barrier = true) AS
        SELECT p.paraphrase_id, p.qbank_id, p.paraphrase, p.search_document
          FROM pmmc_runtime.qbank_paraphrases p
         WHERE EXISTS (SELECT 1 FROM pmmc_runtime.verified_qbank_routing_v4 r
                        WHERE r.qbank_id=p.qbank_id AND r.compiler_run_id=p.run_id)
    """)
    for view in _ONLINE_VIEW_ADDITIONS:
        op.execute(f"ALTER VIEW pmmc_runtime.{view} OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON pmmc_runtime.{view} FROM PUBLIC")
        op.execute(f"GRANT SELECT ON pmmc_runtime.{view} "
                   "TO pmmc_runtime,pmmc_evaluator,pmmc_online")


def downgrade() -> None:
    for view in reversed(_ONLINE_VIEW_ADDITIONS):
        op.execute(f"DROP VIEW pmmc_runtime.{view}")
