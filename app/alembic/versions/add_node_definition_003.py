"""add node_definition and node_entity_definition tables

Revision ID: add_node_definition_003
Revises: add_target_node_description_002
Create Date: 2026-03-14 01:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


revision = 'add_node_definition_003'
down_revision = 'add_target_node_description_002'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'node_definition',
        sa.Column('label', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(length=500), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'node_entity_definition',
        sa.Column('key', sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column('value_type', sqlmodel.sql.sqltypes.AutoString(length=50), nullable=False, server_default='str'),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(length=500), nullable=False),
        sa.Column('example_value', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('is_required', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('node_definition_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('current_timestamp(0)'), nullable=False),
        sa.ForeignKeyConstraint(['node_definition_id'], ['node_definition.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.add_column(
        'relation_definition',
        sa.Column('node_definition_id', sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        'fk_relation_node_definition',
        'relation_definition', 'node_definition',
        ['node_definition_id'], ['id'],
        ondelete='SET NULL',
    )


def downgrade():
    op.drop_constraint('fk_relation_node_definition', 'relation_definition', type_='foreignkey')
    op.drop_column('relation_definition', 'node_definition_id')
    op.drop_table('node_entity_definition')
    op.drop_table('node_definition')
