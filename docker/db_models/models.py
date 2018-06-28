from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.dialects import postgresql


db = SQLAlchemy()


class WebRequest(db.Model):
    """
    The request class for database storage.
    """
    id = db.Column(db.Integer, primary_key=True)
    sha256 = db.Column(db.String(80), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    possible_classification = db.Column(db.String(120), nullable=True)
    ip_address = db.Column(postgresql.INET)


class Submission(db.Model):
    """
    The submission class for database storage.
    """
    id = db.Column(db.Integer, primary_key=True)
    sha256 = db.Column(db.String(80), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    classification = db.Column(db.String(120), nullable=True)
    possible_classification = db.Column(db.String(120), nullable=True)
    ip_address = db.Column(postgresql.INET)
    status = db.Column(db.String(120), nullable=True)


TABLES = [WebRequest, Submission]


def setup_database():
    """
    Sets up the DB.

    :return:  Nothing.
    """
    db.create_all()
    # engine = sqlalchemy.create_engine(databaseURI)
    # for table in TABLES:
    #     if not table_exists(databaseURI, table.__tablename__):
    #         table.__table__.create(engine)


def clear_database():
    """
    Sets up the DB.

    :return:  Nothing.
    """
    db.drop_all()
    setup_database()
    # engine = sqlalchemy.create_engine(databaseURI)
    # for table in TABLES:
    #     if table_exists(databaseURI, table.__tablename__):
    #         table.__table__.drop(engine)


def database_is_empty(databaseURI):
    """
    Checks to see if the database is empty.

    :param databaseURI:  The connection URI.
    :return: Boolean
    """
    engine = sqlalchemy.create_engine(databaseURI)
    table_names = sqlalchemy.inspect(engine).get_table_names()
    is_empty = table_names == []
    return is_empty


def table_exists(databaseURI, name):
    """

    :param databaseURI:  The connection URI.
    :param name:  The name of the table to check.
    :return: Boolean
    """
    engine = sqlalchemy.create_engine(databaseURI)
    ret = engine.dialect.has_table(engine, name)
    return ret