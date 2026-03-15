"""add is_global to relation_definition

Revision ID: add_is_global_004
Revises: add_node_definition_003
Create Date: 2026-03-15 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'add_is_global_004'
down_revision = 'add_node_definition_003'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'relation_definition',
        sa.Column('is_global', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )


def downgrade():
    op.drop_column('relation_definition', 'is_global')
