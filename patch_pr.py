import re

with open('src/services/image-generation.ts', 'r') as f:
    content = f.read()

content = content.replace("ollamaUrl: this.plugin.ollamaUrl", "baseUrl: this.plugin.ollamaUrl")

with open('src/services/image-generation.ts', 'w') as f:
    f.write(content)
