cd /malgazer
sleep 10
export FLASK_APP=docker.api.api

if [ $MALGAZER_RUN_ENV == "dev" ]
then
    export FLASK_ENV=development
elif [ $MALGAZER_RUN_ENV == 'prod' ]
then
    export FLASK_ENV=production
fi

if [ $MALGAZER_RUN_ENV == "dev" ]
then
    flask run --host=0.0.0.0 --port=8888
elif [ $MALGAZER_RUN_ENV == 'prod' ]
then
    gunicorn -w $MALGAZER_API_WORKERS -t MALGAZER_API_THREADS -k gthread -b 0.0.0.0:8888 docker.api.api:app
fi
