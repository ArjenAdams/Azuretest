/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      colors: {
        "primary" : {
          100 : "#cdfaef",
          200 : "#9cf3e0",
          300 : "#62e6ce",
          DEFAULT: "#2fc8b1",
          400 : "#2fc8b1",
          500 : "#19b39e",
          600 : "#119082",
          700 : "#127369",
          800 : "#135c55",
          900 : "#154c47",
        },
        "secondary" : {
          100 : "#e0e9ff",
          200 : "#c7d5fe",
          300 : "#a5b9fc",
          400 : "#8193f8",
          500 : "#636ff1",
          600 : "#4647e5",
          700 : "#3a38ca",
          800 : "#3030a3",
          DEFAULT: "#2d2f7f",
          900 : "#2d2f7f",
        }
      }
    },
  },
  plugins: [],
}

