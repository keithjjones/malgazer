from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.dialects import postgresql
from binascii import hexlify
import os


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


class User(db.Model):
    """
    The user class for database storage.
    """
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    permissions = db.Column(db.String(100), nullable=False)
    api_key = db.Column(db.String(200), nullable=True)
    registration = db.Column(db.DateTime, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)


class SystemLog(db.Model):
    """
    The class for the system logs for database storage.
    """
    id = db.Column(db.Integer, primary_key=True)
    node = db.Column(db.String(100), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    message = db.Column(db.DateTime, nullable=False)


class UserLog(db.Model):
    """
    The class for the user logs for database storage.
    """
    id = db.Column(db.Integer, primary_key=True)
    node = db.Column(db.String(100), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    message = db.Column(db.DateTime, nullable=False)
    ip_address = db.Column(postgresql.INET, nullable=False)


TABLES = [WebRequest, Submission, User]


def generate_api_key(length=60):
    key = hexlify(os.urandom(length)).upper()
    return key.decode()


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