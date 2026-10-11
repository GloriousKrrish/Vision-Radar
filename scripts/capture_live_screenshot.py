import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        print("Navigating to http://localhost:8080...")
        await page.goto("http://localhost:8080", wait_until="networkidle")
        await asyncio.sleep(2)

        # Ensure video is playing or paused cleanly
        screenshot_path = r"C:\Users\admin\.gemini\antigravity-ide\brain\00783d75-5a91-490c-9a6b-cd8ed0c3cc8a\live_repair_verified.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to {screenshot_path}")

        # Check DOM elements
        header_text = await page.locator("header").inner_text()
        print(f"Header: {header_text}")

        inspector_text = await page.locator(".card >> text=Track Inspector").locator("xpath=..").inner_text()
        print(f"Inspector: {inspector_text}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
