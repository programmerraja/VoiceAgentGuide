/* Minimal markdown renderer: headings, lists, quotes, code, links, emphasis.
   Input is escaped first, so note content can never inject HTML. */
(function (global) {
  const escapeHtml = (text) =>
    text.replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));

  function inline(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\W)\*([^*]+)\*/g, "$1<em>$2</em>")
      .replace(
        /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
  }

  function render(markdown) {
    if (!markdown || !markdown.trim()) return "";
    const lines = escapeHtml(markdown).split("\n");
    const out = [];
    let listType = null;
    let inCode = false;
    let paragraph = [];

    const flushParagraph = () => {
      if (paragraph.length) {
        out.push(`<p>${inline(paragraph.join(" "))}</p>`);
        paragraph = [];
      }
    };
    const closeList = () => {
      if (listType) {
        out.push(`</${listType}>`);
        listType = null;
      }
    };

    for (const line of lines) {
      if (line.trim().startsWith("```")) {
        flushParagraph();
        closeList();
        out.push(inCode ? "</code></pre>" : "<pre><code>");
        inCode = !inCode;
        continue;
      }
      if (inCode) {
        out.push(line + "\n");
        continue;
      }

      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
      const quote = line.match(/^>\s?(.*)$/);

      if (!line.trim()) {
        flushParagraph();
        closeList();
      } else if (heading) {
        flushParagraph();
        closeList();
        const level = Math.min(heading[1].length + 2, 6);
        out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      } else if (bullet || numbered) {
        flushParagraph();
        const wanted = bullet ? "ul" : "ol";
        if (listType !== wanted) {
          closeList();
          out.push(`<${wanted}>`);
          listType = wanted;
        }
        out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      } else if (quote) {
        flushParagraph();
        closeList();
        out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
      } else {
        paragraph.push(line.trim());
      }
    }

    flushParagraph();
    closeList();
    if (inCode) out.push("</code></pre>");
    return out.join("");
  }

  global.Markdown = { render, escapeHtml };
})(window);
