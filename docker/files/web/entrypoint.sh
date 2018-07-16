cd /malgazer
sleep 10
export FLASK_APP=docker.web.web

if [ $MALGAZER_RUN_ENV == "dev" ]
then
    export FLASK_ENV=development
elif [ $MALGAZER_RUN_ENV == 'prod' ]
then
    export FLASK_ENV=production
fi

if [ $MALGAZER_RUN_ENV == "dev" ]
then
    flask run --host=0.0.0.0 --port=8080
elif [ $MALGAZER_RUN_ENV == 'prod' ]
then
    gunicorn -w $MALGAZER_WEB_WORKERS -t $MALGAZER_WEB_THREADS -k gthread -b 0.0.0.0:8080 docker.web.web:app
fi