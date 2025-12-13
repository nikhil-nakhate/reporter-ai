"""
Service for fetching and extracting article content from URLs.
"""
import logging
import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ArticleService:
    """Service for fetching article content from URLs."""
    
    def __init__(self):
        """Initialize the article service."""
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def fetch_article(self, url: str) -> str:
        """
        Fetch and extract article content from a URL.
        
        Args:
            url: URL of the article to fetch
            
        Returns:
            Extracted article text content
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to fetch article: HTTP {response.status}")
                    
                    html = await response.text()
                    content = self._extract_content(html)
                    
                    if not content or len(content.strip()) < 100:
                        raise Exception("Could not extract sufficient content from article")
                    
                    logger.info(f"Successfully extracted {len(content)} characters from {url}")
                    return content
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching article: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching article: {e}")
            raise
    
    def _extract_content(self, html: str) -> str:
        """
        Extract main article content from HTML.
        
        Args:
            html: HTML content of the page
            
        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find article content using common selectors
        article_selectors = [
            'article',
            '[role="article"]',
            '.article-content',
            '.article-body',
            '.post-content',
            '.entry-content',
            'main',
            '.content',
            '#content',
            '.story-body',
            '.article-text'
        ]
        
        article_content = None
        for selector in article_selectors:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        # If no article tag found, try to find the largest text block
        if not article_content:
            # Get all paragraphs
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Find the container with most paragraphs
                parent = paragraphs[0].parent
                if parent:
                    article_content = parent
        
        if article_content:
            # Extract text
            text = article_content.get_text(separator='\n\n', strip=True)
            # Clean up excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        else:
            # Fallback: get all text from body
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n\n', strip=True)
                text = re.sub(r'\n{3,}', '\n\n', text)
                return text.strip()
        
        return ""

