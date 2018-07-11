from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.dialects import postgresql
from binascii import hexlify
import os
from passlib.hash import bcrypt
from flask_login import UserMixin


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
    email = db.Column(db.Text, nullable=False, index=True, unique=True)
    password = db.Column(db.Text, nullable=False)
    config = db.Column(postgresql.JSON, nullable=True)
    api_key = db.Column(db.Text, nullable=True, index=True)
    registration = db.Column(db.DateTime, nullable=False)
    activated = db.Column(db.Boolean, default=False)
    activated_date = db.Column(db.DateTime, nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    last_login_ip = db.Column(postgresql.INET, nullable=True)

    def __init__(self, *args, **kwargs):
        if 'password' in kwargs:
            kwargs['password'] = bcrypt.encrypt(kwargs['password'])
        super(User, self).__init__(*args, **kwargs)

    def validate_password(self, password):
        login = bcrypt.verify(password, self.password)
        return login

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        # return self.activated
        return True

    def get_id(self):
        return "{0}".format(self.id)

    @property
    def is_anonymous(self):
        return False




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