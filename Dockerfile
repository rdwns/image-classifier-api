FROM python:3.8-buster as builder

ENV PYTHONBUFFERED 1

WORKDIR /app

#Setting up a seperate user for the app, for security


RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

COPY requirements.txt .
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

#--------------------------------------------------------------------------------------------------

FROM python:3.8-buster

ENV USER api
ENV UID 61000
ENV GROUP api
ENV GID 61000

RUN groupadd -g $GID $GROUP
RUN useradd -g $GID -l -m -s /bin/false -u $UID $USER

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

WORKDIR /app
COPY --chown=$UID:$GID ./ .

RUN pip3 install --upgrade pip --no-cache /wheels/*
USER $UID:$GID

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=3s \
  CMD curl -f http://0.0.0.0:80/healthcheck || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]