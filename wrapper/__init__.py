"""
Product Wrapper Layer for Nifty Dynamic Allocator
Includes:
- PortfolioParser (parser.py)
- AllocationRouter (router.py)
"""

from .parser import PortfolioParser
from .router import AllocationRouter

__all__ = [
    "PortfolioParser",
    "AllocationRouter",
]
