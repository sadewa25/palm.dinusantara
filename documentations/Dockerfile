FROM node:20.18.1-alpine3.21 AS build

WORKDIR /app

RUN npm install -g pnpm

COPY . .

RUN pnpm install
RUN pnpm run build

FROM node:20.18.1-alpine3.21 AS runtime
WORKDIR /app

# Install pnpm in the runtime image
RUN npm install -g pnpm

# Copy package.json and pnpm-lock.yaml
COPY package.json pnpm-lock.yaml ./

# Install production dependencies
RUN pnpm install --prod

# Copy built application
COPY --from=build /app/dist ./dist

ENV HOST=0.0.0.0
ENV PORT=3002
EXPOSE 3002

CMD ["pnpm", "preview"]