"""Render numpydoc nodes missing from sphinx-markdown-builder."""

from docutils import nodes
from sphinx_markdown_builder.escape import escape_markdown_chars
from sphinx_markdown_builder.translator import MarkdownTranslator


class GitHubMarkdownTranslator(MarkdownTranslator):
    """Keep parameter types, citations and signature markers in Markdown."""

    def visit_Text(self, node):
        text = node.astext().replace("\r", "")
        if not self.status.preserve_line_breaks:
            text = text.replace("\n", " ")
        if self.status.escape_text:
            text = escape_markdown_chars(text)
            text = text.replace("[", r"\[").replace("]", r"\]")
        self.add(text)

    def visit_classifier(self, node):
        self.add(" — ")

    def visit_abbreviation(self, node):
        pass

    def visit_citation(self, node):
        self.ensure_eol(2)
        for anchor in node.get("ids", []):
            self._add_anchor(anchor)
        self.ensure_eol(2)

    def depart_citation(self, node):
        self.ensure_eol(2)

    def visit_label(self, node):
        if isinstance(node.parent, nodes.citation):
            self.add("[")
        else:
            super().visit_label(node)

    def depart_label(self, node):
        if isinstance(node.parent, nodes.citation):
            self.add("]")
        else:
            super().depart_label(node)
