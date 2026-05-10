/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'agri-bg':      '#0d1117',
        'agri-surface': '#161b22',
        'agri-border':  '#21262d',
        'agri-green':   '#58d68d',
        'agri-red':     '#e74c3c',
        'agri-yellow':  '#f39c12',
        'agri-text':    '#e6edf3',
        'agri-muted':   '#8b949e',
      },
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['DM Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}