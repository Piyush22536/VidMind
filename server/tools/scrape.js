import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { triggerYoutubeVideoScrape } from '../brightdata.js';

export const triggerScrapeTool = tool(
  async ({ url }) => {
    console.log('[agent] triggering scrape for', url);
    const snapshotId = await triggerYoutubeVideoScrape(url);
    return JSON.stringify({
      snapshot_id: snapshotId,
      message: 'Scrape triggered. The video will be available in ~10 seconds via the webhook.',
    });
  },
  {
    name: 'trigger_youtube_scrape',
    description: `
      Trigger scraping of a YouTube video by URL.
      IMPORTANT: Always call check_video_indexed first to avoid re-scraping already indexed videos.
      The scrape takes ~7-10 seconds; a webhook will index the transcript automatically.
    `,
    schema: z.object({
      url: z.string().describe('Full YouTube URL'),
    }),
  }
);