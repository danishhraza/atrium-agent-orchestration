import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renders what an agent wrote: headings, lists, tables, links, images, code.
 *  react-markdown never injects raw HTML, so model output cannot run script in the page. */
export default function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer noopener" />,
          img: ({ node: _node, ...props }) => <img {...props} loading="lazy" referrerPolicy="no-referrer" />,
          table: ({ node: _node, ...props }) => <div className="md-table"><table {...props} /></div>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
