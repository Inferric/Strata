FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL=http://localhost:8000
ARG VITE_MLFLOW_URL=http://localhost:5000
ARG VITE_PREFECT_URL=http://localhost:4200
ENV VITE_API_URL=${VITE_API_URL}
ENV VITE_MLFLOW_URL=${VITE_MLFLOW_URL}
ENV VITE_PREFECT_URL=${VITE_PREFECT_URL}
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
