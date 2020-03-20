import api_module
from waitress import serve

app = api_module.create_app()
app.run(host='0.0.0.0', port=8081)

## Run with waitress wsgi server
#serve(app, host='0.0.0.0', port=8081)   