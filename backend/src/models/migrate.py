from sqlalchemy import inspect

from src.models.tables import metadata


def ensure_columns(engine) -> list[str]:
    """Add columns introduced after a database volume was first created.

    ``metadata.create_all`` only creates missing tables, so an existing
    deployment would keep the old shape. This inspects each table and adds the
    missing nullable columns in place; it is safe to run on every start.
    """
    added = []
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table in metadata.sorted_tables:
            existing = {column['name'] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:
                    # Never guess a value for a required column; surface the
                    # mismatch instead of writing a half-migrated table.
                    raise RuntimeError(f'{table.name}.{column.name} 缺少非空列，请先迁移数据库。')
                kind = column.type.compile(dialect=engine.dialect)
                connection.exec_driver_sql(
                    f'ALTER TABLE {table.name} ADD COLUMN {column.name} {kind}'
                )
                added.append(f'{table.name}.{column.name}')
    return added
