/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.html', './legal/*.html', './assets/*.js'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#09080c', 900: '#0e0d12', 850: '#131216', 800: '#18171c', 700: '#222124', 600: '#2e2d33' },
        amber: { DEFAULT: '#f0b232', soft: '#f6cd6e' },
        signal: { DEFAULT: '#23a559', soft: '#4fd88a' },
        bayer: { r: '#e5484d', g: '#3fb950', b: '#4c8dff' },
        mute: { DEFAULT: '#a6a3ad', strong: '#cfcdd4' }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', '"Segoe UI"', 'Roboto', 'sans-serif'],
        display: ['"Space Grotesk"', 'Inter', 'system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', '"Cascadia Code"', 'Consolas', '"Liberation Mono"', 'monospace']
      },
      maxWidth: { page: '72rem' }
    }
  },
  plugins: []
};
