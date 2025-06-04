import json
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_URL = "https://ui.camped.academy"
COMPONENTS_PATH = "/docs/components/"

# Exact list of component names
COMPONENT_NAMES = [
    "Accordion", "Alert", "Aspect Ratio", "Avatar", "Badge", "Bento Grid", "Breadcrumb",
    "Button", "Card", "Carousel", "Collapsible", "Command", "Context Menu", "Data Table",
    "Dropdown Menu", "Form", "Hover Card", "Label", "Menubar", "Navigation Menu",
    "Pagination", "Progress", "Resizable", "Scroll Area", "Separator", "Skeleton",
    "Slider", "Stepper", "Sticky Scroll Reveal", "Switch", "Table", "Tabs",
    "Toggle", "Toggle Group", "Tooltip", "Tree View", "Inputs", "Input", "Input OTP",
    "Password Input", "Multi Select", "Select", "Textarea", "Tag Input", "Calendar",
    "Checkbox", "Combobox", "Date Picker", "Radio Group", "File Upload", "Text", "Text",
    "Typewriter Effect", "Modals", "Alert Dialog", "Sheet", "Popover", "Dialog"
]


def slugify(name: str) -> str:
    """
    Convert a component name into the URL slug (lowercase, spaces → hyphens).
    """
    return name.lower().replace(" ", "-")


def build_component_urls():
    """
    Build (component_name, full_url) tuples based on COMPONENT_NAMES.
    """
    urls = []
    for name in COMPONENT_NAMES:
        if not name.strip():
            continue
        slug = slugify(name)
        full_url = f"{BASE_URL}{COMPONENTS_PATH}{slug}"
        urls.append((name, full_url))
    return urls


def scrape_all_components():
    """
    Uses Playwright to load each component page, click all "Code" buttons so that
    every example's <pre> block is visible, then run a JS snippet to extract:
      - page_title (the <h1> text)
      - sections: for each <h2> (Installation, Usage, Examples, etc.)
          • pre_snippets: any <pre> before the first <h3>
          • subsections: for each <h3> under that <h2> (like Primary, Secondary, etc.),
            collect all <pre> tags between that <h3> and the next <h3> or next <h2>.
    """
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()

        for comp_name, comp_url in build_component_urls():
            print(f"⏳  Scraping '{comp_name}' → {comp_url}")
            try:
                # 1) Navigate to the page and wait for DOM content to load
                page.goto(comp_url, wait_until="domcontentloaded", timeout=30000)

                # 2) Wait up to 10s for “Examples” H2 to appear, if it exists.
                #    This ensures React has mounted the example toggles.
                try:
                    page.wait_for_selector("h2:has-text('Examples')", timeout=10000)
                except PlaywrightTimeoutError:
                    pass  # no Examples section? we’ll still click any Code buttons that exist.

                # 3) Click all “Code” buttons on the page so that code blocks become visible.
                #    We loop because locator.click() clicks only the first match by default.
                code_buttons = page.locator("button:has-text('Code')")
                count = code_buttons.count()
                for i in range(count):
                    try:
                        code_buttons.nth(i).click()
                    except Exception:
                        # sometimes a button might not be attached yet—ignore errors
                        pass

                # 4) Now run a single JS snippet in the page’s context to extract <h1>, <h2>, <h3>, and <pre>.
                data = page.evaluate(
                    """() => {
                        const result = {};
                        // (a) Grab <h1> as the page_title
                        const h1 = document.querySelector('h1');
                        result.page_title = h1 ? h1.innerText.trim() : null;

                        // (b) Find every <h2> on the page
                        const h2s = Array.from(document.querySelectorAll('h2'));
                        const sections = [];

                        h2s.forEach((h2, idx) => {
                            const sectionTitle = h2.innerText.trim();
                            const nextH2 = h2s[idx + 1] || null;

                            // Collect siblings from h2.nextSibling up until nextH2
                            const siblings = [];
                            let node = h2.nextSibling;
                            while (node && node !== nextH2) {
                                siblings.push(node);
                                node = node.nextSibling;
                            }

                            // (i) pre_snippets: any <pre> found in siblings BEFORE the first <h3>
                            const preBeforeH3 = [];
                            let encounteredH3 = false;
                            for (const sib of siblings) {
                                if (sib.nodeType === Node.ELEMENT_NODE && sib.tagName.toLowerCase() === 'h3') {
                                    encounteredH3 = true;
                                    break;
                                }
                                if (sib.nodeType === Node.ELEMENT_NODE) {
                                    if (sib.tagName.toLowerCase() === 'pre') {
                                        preBeforeH3.push(sib.innerText);
                                    }
                                    // Also check nested <pre>
                                    sib.querySelectorAll('pre').forEach(p => preBeforeH3.push(p.innerText));
                                }
                            }

                            // (ii) Find all <h3> within these same siblings
                            const h3s = siblings.filter(el => 
                                el.nodeType === Node.ELEMENT_NODE && el.tagName.toLowerCase() === 'h3'
                            );

                            // (iii) For each H3, collect <pre> tags between it and the next <h3> or next <h2>
                            const subsections = h3s.map((h3, h3idx) => {
                                const subTitle = h3.innerText.trim();
                                const nextH3 = h3s[h3idx + 1] || nextH2;

                                // Walk siblings from h3.nextSibling until nextH3 or nextH2
                                const codes = [];
                                let node2 = h3.nextSibling;
                                while (node2 && node2 !== nextH3) {
                                    if (node2.nodeType === Node.ELEMENT_NODE) {
                                        if (node2.tagName.toLowerCase() === 'pre') {
                                            codes.push(node2.innerText);
                                        }
                                        node2.querySelectorAll('pre').forEach(p => codes.push(p.innerText));
                                    }
                                    node2 = node2.nextSibling;
                                }
                                return { subsection: subTitle, code_snippets: codes };
                            });

                            sections.push({
                                section: sectionTitle,
                                pre_snippets: preBeforeH3,
                                subsections: subsections
                            });
                        });

                        result.sections = sections;
                        return result;
                    }
                    """
                )

                # Attach the URL/name and error=null
                results.append({
                    "name": comp_name,
                    "url": comp_url,
                    "page_title": data["page_title"],
                    "sections": data["sections"],
                    "error": None
                })
                print(f"  ✅  Collected {len(data['sections'])} sections.")

            except Exception as e:
                print(f"  ❌  Error scraping '{comp_name}': {e}")
                results.append({
                    "name": comp_name,
                    "url": comp_url,
                    "page_title": None,
                    "sections": [],
                    "error": str(e)
                })

        browser.close()

    # 5) Write everything to JSON
    with open("campedui_components_playwright_codeclick.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n🎉 Done! See campedui_components_playwright_codeclick.json for the output.")


if __name__ == "__main__":
    scrape_all_components()
