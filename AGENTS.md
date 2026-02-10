# Agent Guidelines for TypeScript Console Editor with WebGPU

This is a TypeScript Console Editor with WebGPU support, built with Bun and Vite. It features a Monaco editor for writing TypeScript code that gets transpiled and executed, with a split-pane output showing WebGPU canvas rendering (top) and console output (bottom).

## Development Commands

### Package Manager

- Use `bun` as the package manager for all operations
- Install dependencies: `bun install`
- Add new dependencies: `bun add <package>`
- Add dev dependencies: `bun add -d <package>`

### Development Server

- Start dev server: `bun run dev`
- Build for production: `bun run build`
- Preview production build: `bun run preview`

### Code Quality

- Type check: `bun run typecheck`
- Lint code: `bun run lint`
- Format code: `bun run format`

## Project Structure

```
/
├── src/
│   ├── main.ts              # Application entry point - initializes editor and WebGPU
│   ├── editor.ts            # Monaco editor setup, TypeScript transpilation, code execution
│   └── style.css           # Global styles
├── backup/                  # Backup of previous versions
│   ├── src/
│   └── webgpu/             # Old WebGPU implementation files
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── vite.config.ts          # Vite configuration
├── index.html              # HTML entry point
└── AGENTS.md               # This file
```

## Architecture

### Editor (`src/editor.ts`)

- Monaco editor integration for TypeScript editing
- TypeScript transpilation using the `typescript` package (not Monaco's worker)
- Code execution in a sandboxed `Function` context
- WebGPU context injection (device, context, format)
- Console output capture and display
- Render loop management for WebGPU

### Layout

- **Left pane**: Monaco TypeScript editor
- **Right pane (top)**: WebGPU canvas for rendering
- **Right pane (bottom)**: Console output
- **Header**: Title with "Compile & Run" button

## Code Execution Flow

1. User writes TypeScript code in Monaco editor
2. User clicks "Compile & Run" button
3. Code is transpiled using `ts.transpileModule()`
4. TypeScript errors are shown if present
5. WebGPU is initialized (adapter, device, context)
6. Code is executed with WebGPU context injected
7. If code returns a renderer with `render()` method, a render loop starts
8. Console output is captured and displayed

## Code Style Guidelines

### TypeScript

- Use strict TypeScript configuration
- Prefer explicit type annotations for function parameters and return types
- Use `interface` for object shapes, `type` for unions/primitives

### Import Organization

```typescript
// External libraries
import * as monaco from 'monaco-editor';
import * as ts from 'typescript';

// Relative imports for local files
import './style.css';
```

### Naming Conventions

- **Files**: kebab-case (e.g., `editor.ts`, `style.css`)
- **Classes/PascalCase**: `Renderer`
- **Functions/Variables**: camelCase (e.g., `executeCode`, `logToConsole`)
- **Constants**: UPPER_SNAKE_CASE for true constants

### Error Handling

```typescript
// Always handle errors in async operations
try {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('Failed to request WebGPU adapter');
  }
} catch (error) {
  console.error('WebGPU initialization failed:', error);
  throw error;
}
```

## Key Implementation Details

### TypeScript Transpilation

- Uses `typescript.transpileModule()` for reliable transpilation
- Strips `import` and `export` statements for Function execution
- Injects WebGPU context (`device`, `context`, `format`) as parameters

### WebGPU Integration

- Checks for `navigator.gpu` support
- Requests adapter and device
- Configures canvas context with preferred format
- Injects context into user code via `new Function()`
- Manages render loop with `requestAnimationFrame`

### Console Capture

- Intercepts `console.log`, `console.error`, `console.warn`
- Displays output in styled console div
- Restores original console methods after execution

### UI Components

- Monaco editor with TypeScript support
- Canvas element for WebGPU rendering
- Console output with log/error/warn styling
- Error banner with line number highlighting
- Compile button with gradient styling

## Dependencies

### Production

- `monaco-editor`: Code editor
- `@monaco-editor/loader`: Monaco loader
- `typescript`: TypeScript compiler for transpilation

### Development

- `vite`: Build tool
- `@types/node`: Node.js types
- `typescript`: TypeScript compiler

## Browser Compatibility

- Requires modern browser with WebGPU support (Chrome/Edge)
- Uses ES2022 target for transpiled code
- Uses ES modules

## Security Considerations

- User code executes in a sandboxed `Function` context
- WebGPU device is created fresh for each execution
- No sensitive data is exposed to user code

## Backup Files

Previous implementations are backed up in `/backup/`:

- `src/editor.ts.*.bak`: Various editor implementations
- `src/main.ts.*.bak`: Previous main.ts versions
- `webgpu/`: Old WebGPU renderer, shaders, and utilities
