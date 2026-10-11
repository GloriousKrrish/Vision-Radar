import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        await page.goto("http://localhost:8080", wait_until="networkidle")
        await asyncio.sleep(2)

        # Let's inspect the page state via evaluate
        res = await page.evaluate("""() => {
            return {
                title: document.title,
                header: document.querySelector('header b')?.textContent,
                statusBadge: document.querySelector('header span')?.textContent || document.querySelector('header')?.innerText,
                status: document.querySelector('.card')?.innerText
            };
        }""")
        print("Page state evaluated successfully.")

        # Save screenshot
        path = r"C:\Users\admin\.gemini\antigravity-ide\brain\00783d75-5a91-490c-9a6b-cd8ed0c3cc8a\job_inspected_verified.png"
        await page.screenshot(path=path, full_page=True)
        print(f"Screenshot saved to {path}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
