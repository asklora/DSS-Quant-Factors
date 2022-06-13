from utils.packages.sql.src.models.bases import Base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP, JSON, NUMERIC, BIGINT
from sqlalchemy import Column, ForeignKey


# class NewTable(Base):
#     __tablename__ = 'new_table'
#
#     id = Column(INTEGER, primary_key=True)