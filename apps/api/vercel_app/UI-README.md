# Calculator UI - Next.js Application

A modern, fully-functional calculator web application built with Next.js, React, and Tailwind CSS.

## Features

- **Basic Operations**: Addition, subtraction, multiplication, division
- **Advanced Functions**: Square root, power, modulo operations
- **Modern UI**: Dark theme with smooth animations and responsive design
- **Calculation History**: View previous calculations
- **Keyboard Support**: Designed for both mouse and keyboard input
- **Production Ready**: Optimized for deployment on Vercel

## Project Structure

```
vercel_app/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main calculator page
│   └── globals.css         # Global styles
├── components/
│   ├── Calculator.tsx      # Main calculator logic
│   ├── Display.tsx         # Display component
│   └── Keypad.tsx          # Keypad buttons
├── package.json            # Dependencies
├── tailwind.config.js      # Tailwind configuration
├── next.config.js          # Next.js configuration
├── tsconfig.json           # TypeScript configuration
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Node.js 18+ or higher
- pnpm, npm, yarn, or bun package manager

### Installation

1. Navigate to the calculator app directory:
   ```bash
   cd apps/api/vercel_app
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   pnpm install
   # or
   yarn install
   ```

### Development

Run the development server:

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the calculator.

## Building for Production

Build the application:

```bash
npm run build
```

Start the production server:

```bash
npm run start
```

## Deployment

### Vercel Deployment

This application is optimized for deployment on Vercel:

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Vercel will automatically detect the Next.js application and deploy it

### Docker Deployment

You can also containerize this application:

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

## Usage

### Calculator Operations

- **Number buttons**: Click to enter digits
- **Decimal point (.)**: Add decimal to numbers
- **Operations (+, −, ×, ÷)**: Perform arithmetic operations
- **Equals (=)**: Calculate the result
- **Clear (C)**: Reset the calculator
- **Backspace (⌫)**: Delete the last digit
- **Toggle Sign (±)**: Make numbers positive or negative
- **Square Root (√)**: Calculate square root
- **Power (^)**: Raise to a power
- **Modulo (%)**: Get remainder

## Components

### Calculator.tsx
Main component that manages calculator state and logic. Handles:
- Number input
- Operation handling
- Calculation logic
- History tracking

### Display.tsx
Shows the current value and previous calculation history.

### Keypad.tsx
Renders all calculator buttons with proper accessibility attributes.

## Styling

The application uses Tailwind CSS with a custom dark theme:
- **Primary**: Blue (#3b82f6)
- **Accent**: Purple (#8b5cf6)
- **Success**: Green (#10b981)
- **Error**: Red (#ef4444)
- **Background**: Dark Slate (#0f172a)

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- Optimized bundle size
- Fast initial page load
- Smooth animations with CSS transitions
- Responsive design for all screen sizes

## Accessibility

- ARIA labels on all buttons
- Keyboard-friendly interface
- Color contrast compliant (WCAG AA)
- Semantic HTML structure

## License

MIT
