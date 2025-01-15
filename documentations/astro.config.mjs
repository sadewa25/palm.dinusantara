import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import icon from "astro-icon";
import { defineConfig } from "astro/config";
import partytown from '@astrojs/partytown'

// https://astro.build/config
export default defineConfig({
  site: "https://palm.dinusantara.id",
  integrations: [
    tailwind(),
    mdx(),
    sitemap(),
    icon(),
    partytown({
      config: {
        forward: ["dataLayer.push"],
      },
    }),
  ],
  server: {
    port: 3002,
  },
  output: "static",
});
