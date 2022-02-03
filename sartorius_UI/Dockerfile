FROM python:3.8-slim

WORKDIR /app

COPY . /app/

RUN  python -m pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["app.py"]


