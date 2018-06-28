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


def setup_database(databaseURI):
    engine = sqlalchemy.create_engine(databaseURI)
    try:
        Submission.__table__.create(engine)
    except:
        pass
    try:
        WebRequest.__table__.create(engine)
    except:
        pass