# research_agent/tools/web_scraper_tool.py
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import logging
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup # For parsing HTML if needed

logger = logging.getLogger(__name__)

class WebScraperTool:
    async def _scrape_single_url_to_markdown(self, url: str, project_id: str) -> str:
        logger.info(f"[{project_id}] WebScraperTool: Attempting to scrape URL: {url}")
        try:
            browser_conf = BrowserConfig(
                headless=True,
                user_agent="ResearchAgent/1.0 (compatible; Mozilla/5.0)" # Be a good citizen
            )
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.35, threshold_type="fixed") # Slightly lower threshold
            )
            run_conf = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=md_generator,
                timeout_seconds=100 # Slightly longer for potentially complex pages
            )
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await crawler.arun(url=url, config=run_conf)
                
                if result and result.success:
                    if result.markdown and result.markdown.fit_markdown:
                        logger.info(f"[{project_id}] WebScraperTool: Successfully scraped and got fit_markdown for {url}")
                        return result.markdown.fit_markdown
                    elif result.markdown and result.markdown.raw_markdown:
                        logger.warning(f"[{project_id}] WebScraperTool: Using raw_markdown (fit_markdown was empty) for {url}")
                        return result.markdown.raw_markdown
                    else:
                        logger.warning(f"[{project_id}] WebScraperTool: Scrape successful but no markdown content for {url}")
                        return f"WebScraperTool: Scrape successful but no markdown content found for {url}."
                else:
                    err_msg = result.error_message if result else "Unknown crawl error"
                    logger.warning(f"[{project_id}] WebScraperTool: Scrape failed or no result for {url}. Error: {err_msg}")
                    return f"WebScraperTool: Scrape failed for {url}. Error: {err_msg}"
        except Exception as e:
            logger.error(f"[{project_id}] WebScraperTool: Exception during scrape of {url}: {str(e)}", exc_info=True)
            return f"WebScraperTool: Exception scraping {url}: {str(e)}"

    def _parse_duckduckgo_serp(self, html_content: str, project_id: str, max_results: int = 5) -> list[str]:
        """Parses DuckDuckGo HTML to extract result URLs. This is fragile."""
        logger.info(f"[{project_id}] WebScraperTool: Parsing DuckDuckGo SERP HTML.")
        urls = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # DuckDuckGo's structure can change. This selector is an example and likely needs updates.
            # Looking for result links, often within <a> tags with specific classes or parent structures.
            # Example (very hypothetical, inspect actual DDG HTML):
            # result_links = soup.select('div.results article div.result__body h2 a.result__a')
            # A more common pattern for DDG is links with `data-testid="result-title-a"`
            result_links = soup.find_all('a', attrs={'data-testid': 'result-title-a'}, href=True)

            if not result_links: # Fallback to a more generic search for links if specific one fails
                logger.warning(f"[{project_id}] WebScraperTool: Specific DDG selectors failed. Trying generic link search on SERP.")
                # This is very broad and might pick up non-result links
                all_links = soup.find_all('a', href=True)
                # Try to filter for plausible result links (e.g., not internal DDG links, not ads)
                for link in all_links:
                    href = link['href']
                    if href.startswith("http") and "duckduckgo.com" not in href and "duck.com" not in href:
                         # Further filtering might be needed (e.g., based on link text or parent elements)
                        if len(urls) < max_results * 2: # Get a bit more to filter down
                            urls.append(href)
                        else:
                            break
                # If we used this fallback, we might have many irrelevant links.
                # An LLM call to filter these URLs based on relevance to the query might be needed.
                # For now, we'll just take them.
                logger.info(f"[{project_id}] WebScraperTool: Found {len(urls)} potential URLs via generic SERP link search.")


            for link in result_links:
                if link['href']:
                    urls.append(link['href'])
                    if len(urls) >= max_results:
                        break
            
            # Deduplicate while preserving order (if order matters)
            seen = set()
            urls = [x for x in urls if not (x in seen or seen.add(x))]

            logger.info(f"[{project_id}] WebScraperTool: Extracted {len(urls)} URLs from SERP.")
            return urls[:max_results] # Return only up to max_results
        except Exception as e:
            logger.error(f"[{project_id}] WebScraperTool: Error parsing SERP HTML: {e}", exc_info=True)
            return []


    async def search_and_extract_urls(self, query: str, project_id: str, max_search_results: int = 3) -> list[str]:
        """
        Performs a search on DuckDuckGo, scrapes the SERP, and extracts result URLs.
        """
        logger.info(f"[{project_id}] WebScraperTool: Performing search for query: '{query}'")
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}" # Use HTML version for easier parsing
        
        logger.info(f"[{project_id}] WebScraperTool: Scraping SERP URL: {search_url}")
        serp_html = await self._scrape_single_url_to_markdown(search_url, project_id) # Misnomer, it returns HTML for DDG HTML version

        if "Error scraping" in serp_html or "Scrape failed" in serp_html or not serp_html.strip():
            logger.error(f"[{project_id}] WebScraperTool: Failed to scrape SERP for query '{query}'. Content: {serp_html[:200]}")
            return []
        
        # The HTML version of DDG returns HTML, not Markdown. So we parse HTML.
        extracted_urls = self._parse_duckduckgo_serp(serp_html, project_id, max_results=max_search_results)
        return extracted_urls

    async def ascrape_urls_to_markdown(self, urls: list[str], project_id: str) -> dict[str, str]:
        """Asynchronously scrapes a list of URLs and returns their markdown content."""
        logger.info(f"[{project_id}] WebScraperTool: Starting async scrape for {len(urls)} URLs.")
        results = {}
        tasks = [self._scrape_single_url_to_markdown(url, project_id) for url in urls]
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url, content_or_exc in zip(urls, scraped_contents):
            if isinstance(content_or_exc, Exception):
                logger.error(f"[{project_id}] WebScraperTool: Exception for URL {url} in gather: {content_or_exc}")
                results[url] = f"WebScraperTool: Exception during batch scrape for {url}: {content_or_exc}"
            else:
                results[url] = content_or_exc
        logger.info(f"[{project_id}] WebScraperTool: Finished async scrape for {len(urls)} URLs.")
        return results

    def scrape_urls_sync_via_async_wrapper(self, urls: list[str], project_id: str) -> dict[str, str]:
        logger.info(f"[{project_id}] WebScraperTool: Sync wrapper for scraping {len(urls)} URLs.")
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                logger.warning(f"[{project_id}] WebScraperTool: Asyncio loop already running. Using ThreadPoolExecutor for scraping.")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(urls) if urls else 1)) as executor:
                    # Create a new loop for each thread if `asyncio.run` is used inside
                    def run_in_new_loop(async_fn, *args):
                        return asyncio.run(async_fn(*args))
                    
                    future_to_url_map = {}
                    temp_results = {}

                    # Submit scraping tasks
                    url_batches = [urls[i:i + 3] for i in range(0, len(urls), 3)] # Process in smaller batches
                    for batch in url_batches:
                        if not batch: continue
                        future = executor.submit(run_in_new_loop, self.ascrape_urls_to_markdown, batch, project_id)
                        for url_in_batch in batch:
                            future_to_url_map[url_in_batch] = future # Map URL to its batch future

                    # Collect results
                    # This needs to be handled carefully as one future returns a dict for a batch
                    batch_futures_done = {}
                    for url_key, fut in future_to_url_map.items():
                        if fut not in batch_futures_done:
                            try:
                                batch_result_dict = fut.result(timeout=120 * len(fut._kwargs.get('urls', [1]))) # Timeout per batch
                                temp_results.update(batch_result_dict)
                            except Exception as exc:
                                logger.error(f"[{project_id}] WebScraperTool: Batch future failed: {exc}", exc_info=True)
                                # Mark all URLs associated with this future as failed
                                for u, f in future_to_url_map.items():
                                    if f == fut:
                                        temp_results[u] = f"WebScraperTool: Error processing batch for {u}: {exc}"
                            batch_futures_done[fut] = True
                    return temp_results
            else:
                return loop.run_until_complete(self.ascrape_urls_to_markdown(urls, project_id))
        except RuntimeError: # No event loop in current thread
            logger.info(f"[{project_id}] WebScraperTool: No existing event loop, creating new one for scraping.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.ascrape_urls_to_markdown(urls, project_id))
        except Exception as e:
            logger.error(f"[{project_id}] WebScraperTool: General error in scrape_urls_sync: {e}", exc_info=True)
            return {url: f"WebScraperTool: Sync wrapper error for {url}: {e}" for url in urls}