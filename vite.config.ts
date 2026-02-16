import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
	base: '/HelloWebGPU/',
	resolve: {
		alias: {
			"@": resolve(__dirname, "src"),
		},
	},
	server: {
		port: 3000,
		open: true,
	},
	build: {
		target: "esnext",
		outDir: "dist",
		sourcemap: true,
	},
	optimizeDeps: {
		exclude: ["monaco-editor"],
	},
});
