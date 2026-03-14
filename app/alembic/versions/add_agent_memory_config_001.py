"""add agent memory config tables

Revision ID: add_agent_memory_config_001
Revises: 296abd786d4e
Create Date: 2026-03-14 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'add_agent_memory_config_001'
down_revision = '296abd786d4e'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'relation_definition',
        sa.Column('domain_name', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('action_name', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=True),
        sa.Column('relation_type', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(length=500), nullable=False),
        sa.Column('target_node_label', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'entity_definition',
        sa.Column('key', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('value_type', sqlmodel.sql.sqltypes.AutoString(length=50), nullable=False, server_default='str'),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(length=500), nullable=False),
        sa.Column('example_value', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('is_required', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('relation_definition_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.ForeignKeyConstraint(['relation_definition_id'], ['relation_definition.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade():
    op.drop_table('entity_definition')
    op.drop_table('relation_definition')
