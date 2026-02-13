import esbuild from "esbuild";

const isProduction = process.argv.includes("production");
const isWatch = !isProduction;

const context = await esbuild.context({
  entryPoints: ["main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "@codemirror/*"],
  format: "cjs",
  platform: "browser",
  target: "es2020",
  sourcemap: isProduction ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  logLevel: "info",
});

if (isWatch) {
  await context.watch();
  console.log("Watching for changes...");
} else {
  await context.rebuild();
  await context.dispose();
  console.log("Build complete.");
}
