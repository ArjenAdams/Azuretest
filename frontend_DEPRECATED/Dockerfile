# frontend/Dockerfile
# Base image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json before other files
COPY ./package*.json ./

# Install dependencies
RUN npm install --force

# Copy the rest of the frontend code to the container
COPY . /app

# Expose the port React is running on
EXPOSE 3000

# Start the frontend application
CMD ["npm", "start"]
