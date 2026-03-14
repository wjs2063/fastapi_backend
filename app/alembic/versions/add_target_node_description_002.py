"""add target_node_description to relation_definition

Revision ID: add_target_node_description_002
Revises: add_agent_memory_config_001
Create Date: 2026-03-14 00:01:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


revision = 'add_target_node_description_002'
down_revision = 'add_agent_memory_config_001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'relation_definition',
        sa.Column(
            'target_node_description',
            sqlmodel.sql.sqltypes.AutoString(length=500),
            nullable=True,
        ),
    )


def downgrade():
    op.drop_column('relation_definition', 'target_node_description')
