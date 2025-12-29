"""
Command-line interface for MemCtrl
"""

import click
import json
from pathlib import Path

from memctrl import MemoryController
from memctrl.config import get_config


# Global controller instance
_controller = None


def get_controller(user_id: str = None) -> MemoryController:
    """Get or create controller instance"""
    global _controller
    if _controller is None:
        # Use default user or provided user_id
        if user_id is None:
            config = get_config()
            user_file = Path(config.data_dir) / '.current_user'
            if user_file.exists():
                user_id = user_file.read_text().strip()
            else:
                user_id = 'default'
                user_file.parent.mkdir(parents=True, exist_ok=True)
                user_file.write_text(user_id)
        
        _controller = MemoryController(user_id=user_id)
    return _controller


@click.group()
@click.option('--user', '-u', help='User ID')
@click.pass_context
def cli(ctx, user):
    """MemCtrl - Memory management for long-context LLMs"""
    ctx.ensure_object(dict)
    ctx.obj['user_id'] = user


@cli.command()
@click.argument('message')
@click.option('--user', '-u', help='User ID')
def chat(message, user):
    """Chat with the LLM"""
    controller = get_controller(user)
    response = controller.chat(message)
    click.echo(response)


@cli.command()
@click.argument('content')
@click.option('--note', '-n', help='Note about why this is pinned')
@click.option('--user', '-u', help='User ID')
def pin(content, note, user):
    """Pin content to permanent memory"""
    controller = get_controller(user)
    result = controller.pin(content, note=note)
    
    if result['success']:
        click.echo(click.style(result['message'], fg='green'))
        click.echo(f"Chunk ID: {result['chunk_id']}")
    else:
        click.echo(click.style('Failed to pin', fg='red'))


@cli.command()
@click.argument('query')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
@click.option('--user', '-u', help='User ID')
def forget(query, yes, user):
    """Forget chunks matching query"""
    controller = get_controller(user)
    
    # Get matches
    result = controller.forget(query, confirm=not yes)
    
    if not result['success']:
        click.echo(click.style(result['message'], fg='yellow'))
        return
    
    if result.get('confirm_required'):
        # Show matches
        click.echo(f"\n🔍 Found {len(result['matches'])} chunks:\n")
        for i, match in enumerate(result['matches'], 1):
            click.echo(f"{i}. {match['content'][:80]}...")
            click.echo(f"   Created: {match['timestamp']}")
            click.echo()
        
        # Ask for confirmation
        if click.confirm('Forget these chunks?'):
            chunk_ids = [m['chunk_id'] for m in result['matches']]
            result2 = controller.forget_confirmed(chunk_ids)
            click.echo(click.style(result2['message'], fg='green'))
        else:
            click.echo('Cancelled')
    else:
        click.echo(click.style(result['message'], fg='green'))


@cli.command()
@click.argument('content')
@click.option('--user', '-u', help='User ID')
def temp(content, user):
    """Add content to temporary memory (deleted after session)"""
    controller = get_controller(user)
    result = controller.temporary(content)
    
    if result['success']:
        click.echo(click.style(result['message'], fg='green'))


@cli.command()
@click.option('--category', '-c', 
              type=click.Choice(['all', 'pinned', 'session', 'ai_managed']),
              default='all',
              help='Memory category to show')
@click.option('--user', '-u', help='User ID')
def show(category, user):
    """Show current memory state"""
    controller = get_controller(user)
    memory = controller.show_memory(category=category)
    
    click.echo(f"\n🧠 Memory for user: {memory['user_id']}")
    click.echo(f"Timestamp: {memory['timestamp']}\n")
    
    if 'pinned' in memory and memory['pinned']:
        click.echo(click.style('📌 PINNED MEMORIES:', fg='cyan', bold=True))
        for item in memory['pinned']:
            click.echo(f"\n  • {item['content']}")
            if item.get('note'):
                click.echo(f"    Note: {item['note']}")
            click.echo(f"    Pinned: {item['timestamp']}")
        click.echo()
    
    if 'session' in memory and memory['session']:
        click.echo(click.style('💬 CURRENT SESSION:', fg='yellow', bold=True))
        for item in memory['session']:
            click.echo(f"\n  • {item['content']}")
            click.echo(f"    Time: {item['timestamp']}")
        click.echo()
    
    if 'ai_managed' in memory and memory['ai_managed']:
        click.echo(click.style('🤖 AI-MANAGED:', fg='blue', bold=True))
        for item in memory['ai_managed']:
            if item.get('summary'):
                click.echo(f"\n  • {item['summary']}")
                if item.get('importance'):
                    click.echo(f"    Importance: {item['importance']:.2f}")
        click.echo()


@cli.command()
@click.option('--format', '-f', 
              type=click.Choice(['json', 'table']),
              default='table',
              help='Output format')
@click.option('--user', '-u', help='User ID')
def stats(format, user):
    """Show memory usage statistics"""
    controller = get_controller(user)
    stats_data = controller.get_stats()
    
    if format == 'json':
        click.echo(json.dumps(stats_data, indent=2))
    else:
        # Table format
        click.echo(f"\n📊 Memory Statistics - User: {stats_data['user_id']}")
        click.echo(f"Control Mode: {stats_data['control_mode']}\n")
        
        # Tier 0
        tier0 = stats_data['tiers']['tier0']
        util0 = tier0['utilization'] * 100
        bar0 = '█' * int(util0 / 10) + '░' * (10 - int(util0 / 10))
        click.echo(click.style('Tier 0 (GPU Active):', fg='green', bold=True))
        click.echo(f"  {bar0} {util0:.1f}%")
        click.echo(f"  {tier0['current_tokens']:,} / {tier0['max_tokens']:,} tokens")
        click.echo(f"  {tier0['num_chunks']} chunks\n")
        
        # Tier 1
        tier1 = stats_data['tiers']['tier1']
        util1 = tier1['utilization'] * 100
        bar1 = '█' * int(util1 / 10) + '░' * (10 - int(util1 / 10))
        click.echo(click.style('Tier 1 (RAM Compressed):', fg='yellow', bold=True))
        click.echo(f"  {bar1} {util1:.1f}%")
        click.echo(f"  {tier1['current_tokens']:,} / {tier1['max_tokens']:,} tokens")
        click.echo(f"  {tier1['num_chunks']} chunks\n")
        
        # Tier 2
        tier2 = stats_data['tiers']['tier2']
        click.echo(click.style('Tier 2 (Disk Persistent):', fg='blue', bold=True))
        click.echo(f"  Total chunks: {tier2['total_chunks']}")
        click.echo(f"  Pinned: {tier2['pinned_chunks']}")
        click.echo(f"  Sessions: {tier2['total_sessions']}\n")


@cli.command()
@click.option('--format', '-f',
              type=click.Choice(['json', 'text']),
              default='json',
              help='Export format')
@click.option('--output', '-o', help='Output file (default: stdout)')
@click.option('--user', '-u', help='User ID')
def export(format, output, user):
    """Export all user data"""
    controller = get_controller(user)
    data = controller.export_data(format=format)
    
    if output:
        Path(output).write_text(data)
        click.echo(click.style(f'✓ Exported to {output}', fg='green'))
    else:
        click.echo(data)


@cli.command()
def start():
    """Start Gradio web UI"""
    click.echo("Starting Gradio UI...")
    click.echo("(Not implemented yet - coming in Step 6)")
    # TODO: Import and launch Gradio app


@cli.command()
@click.option('--user', '-u', help='User ID')
def clear(user):
    """Clear current session"""
    controller = get_controller(user)
    controller.close_session()
    click.echo(click.style('✓ Session closed', fg='green'))


if __name__ == '__main__':
    cli()